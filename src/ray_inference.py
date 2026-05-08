# Databricks notebook source
# MAGIC %pip install uv

# COMMAND ----------

# MAGIC %sh
# MAGIC uv pip install \
# MAGIC   --no-binary rasterio \
# MAGIC   --excludes ../excludes.txt \
# MAGIC   -r ../requirements.txt

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install Meta's `sam3` package separately
# MAGIC
# MAGIC The PyPI `sam3` (version `0.0.1`) is an unrelated stub. Meta's real
# MAGIC `sam3` is only published as source at
# MAGIC `github.com/facebookresearch/sam3`. We install it after the main
# MAGIC resolution with `--reinstall` (override any stub) and `--no-deps`
# MAGIC (avoid re-resolving samgeo3's `sam3>=0.1.0.20251211` version pin
# MAGIC against Meta's actual version).

# COMMAND ----------

# MAGIC %sh
# MAGIC uv pip install \
# MAGIC   --reinstall \
# MAGIC   --no-deps \
# MAGIC   "sam3 @ git+https://github.com/facebookresearch/sam3.git"

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# Read all widget values. The widgets themselves are declared in the very
# last cell of the notebook (after `dbutils.notebook.exit()`), so a "Run All"
# never re-declares them; they're populated either by a manual run of that
# bottom cell (interactive) or by the job's `base_parameters` block.
text_prompt         = dbutils.widgets.get("text_prompt")
BBOX_MODEL          = dbutils.widgets.get("BBOX_MODEL")
RUN_SEGMENTATION    = dbutils.widgets.get("RUN_SEGMENTATION").lower() == "true"
SEGMENTER_VERSION   = dbutils.widgets.get("SEGMENTER_VERSION")
RUN_DISSOLVE        = dbutils.widgets.get("RUN_DISSOLVE").lower() == "true"
CATALOG             = dbutils.widgets.get("CATALOG")
SCHEMA              = dbutils.widgets.get("SCHEMA")
VOLUME              = dbutils.widgets.get("VOLUME")
image_path          = dbutils.widgets.get("image_path")
SRID                = int(dbutils.widgets.get("SRID"))
IOU_MATCH_THRESHOLD = float(dbutils.widgets.get("IOU_MATCH_THRESHOLD"))

# COMMAND ----------

import os

import numpy as np
import ray
import supervision as sv
import torch

from ray.util.spark import setup_ray_cluster

from samgeo.samgeo2 import SamGeo2
from transformers import Owlv2Processor, Owlv2ForObjectDetection, AutoProcessor, OmDetTurboForObjectDetection, GroundingDinoForObjectDetection
import rasterio
from PIL import Image
from typing import Dict


# COMMAND ----------

# MAGIC %md
# MAGIC # Starting a Ray cluster
# MAGIC
# MAGIC The below configuration is for a 1 driver + 1 worker spark cluster with an A100 GPU on each. See [docs](https://docs.databricks.com/aws/en/machine-learning/ray/ray-create) for more detail.

# COMMAND ----------

# from ray.util.spark import shutdown_ray_cluster
# shutdown_ray_cluster()

# COMMAND ----------

num_cpu_cores_per_worker = 20 # number of cores to allocate to Ray per worker
num_cpus_head_node = 10 # number of cores to allocate to Ray on the head node
num_gpu_per_worker = 1 # number of GPUs to allocate to Ray per worker
num_gpus_head_node = 1 # number of GPUs to allocate to Ray on the head node
min_worker_nodes = 1 # autoscaling minimum number of workers
max_worker_nodes = 1 # autoscaling maximum number of workers

ray_conf = setup_ray_cluster(
  min_worker_nodes=min_worker_nodes,
  max_worker_nodes=max_worker_nodes,
  num_cpus_head_node= num_cpus_head_node,
  num_gpus_head_node= num_gpus_head_node,
  num_cpus_per_node=num_cpu_cores_per_worker,
  num_gpus_per_node=num_gpu_per_worker
  )

# COMMAND ----------

# This class reads GeoTIFFs, scales the bands, and prepares the image arrays.
class PreProcessorStep:
    def __init__(self):
        # This class is lightweight and doesn't load any models.
        pass
    
    def _scale_to_uint8(self, band_data, lower_percentile=2, upper_percentile=98):
        """Scales a single band to uint8 for model consumption."""
        lower = np.percentile(band_data, lower_percentile)
        upper = np.percentile(band_data, upper_percentile)
        clipped_data = np.clip(band_data, lower, upper)
        scaled_data = ((clipped_data - lower) / (upper - lower + 1e-6)) * 255.0
        
        return scaled_data.astype(np.uint8)

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Processes a batch of image paths.
        - Reads each image path from the input batch.
        - Opens the GeoTIFF and creates two versions:
            1. An RGB NumPy array, scaled to uint8 for model input.
            2. The original NumPy array for final mask creation.
        - Extracts and adds CRS and transform metadata to the batch.
        """
        image_paths = batch["image_path"]
        
        rgb_arrays, original_nps, crs_list, transform_list = [], [], [], []

        for path in image_paths:
            with rasterio.open(path) as src:
                # Verify the source file's first three bands really are R, G, B.
                # Both OWLv2 and SAM2 expect RGB uint8; a silent BGR (or RGBN
                # with channels reordered) would degrade detections without
                # any obvious error.
                #
                # ECW orthos commonly ship with `ColorInterp.undefined` on
                # every band -- the metadata simply isn't filled in. In that
                # case we trust the filename convention and proceed; if the
                # metadata IS set and disagrees with R, G, B, fail loudly.
                ci = src.colorinterp
                assert len(ci) >= 3, (
                    f"{path}: only {len(ci)} bands present; need at least 3 for RGB."
                )
                expected = (
                    rasterio.enums.ColorInterp.red,
                    rasterio.enums.ColorInterp.green,
                    rasterio.enums.ColorInterp.blue,
                )
                undefined = rasterio.enums.ColorInterp.undefined
                if all(c == undefined for c in ci[:3]):
                    print(
                        f"{path}: ColorInterp not set on bands 1-3; "
                        f"assuming R, G, B based on file convention."
                    )
                else:
                    assert ci[:3] == expected, (
                        f"{path}: bands 1-3 have colorinterp {ci[:3]}, "
                        f"expected {expected}."
                    )

                image_np = src.read().transpose((1, 2, 0))
                transform = src.transform
                crs = src.crs

                red_scaled = self._scale_to_uint8(image_np[:, :, 0])
                green_scaled = self._scale_to_uint8(image_np[:, :, 1])
                blue_scaled = self._scale_to_uint8(image_np[:, :, 2])
                
                rgb_image_array = np.dstack((red_scaled, green_scaled, blue_scaled))
                
                rgb_arrays.append(rgb_image_array)
                original_nps.append(image_np)
                crs_list.append(str(crs)) # Convert to string for broader compatibility
                transform_list.append(list(transform)) # Convert to list

        batch["rgb_image_array"] = rgb_arrays
        batch["original_image_np"] = original_nps
        batch["crs"] = crs_list
        batch["transform"] = transform_list
        
        return batch

# COMMAND ----------

# This class loads an open-vocabulary detector and finds bounding boxes.
# The model used is selected by the module-level `BBOX_MODEL` constant.
class BBoxPredictorStep:
    # Per-model configuration: HF id, classes, and post-process thresholds.
    _MODELS = {
        "owlv2": {
            "id": "google/owlv2-large-patch14-ensemble",
            "model_cls": Owlv2ForObjectDetection,
            "processor_cls": Owlv2Processor,
            "threshold": 0.15,
            "text_threshold": None,
        },
        "grounding_dino": {
            "id": "IDEA-Research/grounding-dino-base",
            "model_cls": GroundingDinoForObjectDetection,
            "processor_cls": AutoProcessor,
            "threshold": 0.40,
            "text_threshold": 0.30,
        },
        "omdet": {
            "id": "omlab/omdet-turbo-swin-tiny-hf",
            "model_cls": OmDetTurboForObjectDetection,
            "processor_cls": AutoProcessor,
            "threshold": 0.80,
            "text_threshold": None,
        },
    }

    def __init__(self, queries: list):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if BBOX_MODEL not in self._MODELS:
            raise ValueError(
                f"Unknown BBOX_MODEL={BBOX_MODEL!r}. "
                f"Choose from {sorted(self._MODELS)}."
            )
        cfg = self._MODELS[BBOX_MODEL]
        self.model_name = BBOX_MODEL
        self.threshold = cfg["threshold"]
        self.text_threshold = cfg["text_threshold"]
        self.processor = cfg["processor_cls"].from_pretrained(cfg["id"])
        self.model = cfg["model_cls"].from_pretrained(cfg["id"]).to(self.device)
        # Detection prompt is a run-wide constant, so the actor closes over
        # the parsed query list rather than reading it from a per-row column.
        self.queries = list(queries)

        print(f"BBoxPredictorStep initialized: model={self.model_name}, device={self.device}")

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Processes a batch of images to find bounding boxes for ``self.queries``.
        - Converts NumPy image arrays to PIL Images.
        - Runs the configured open-vocabulary detector on each slice.
        - Emits per-image lists of pixel-space boxes, class indices, and
          confidence scores; the index->label mapping happens in
          ``BoxGeometryStep``.
        """
        queries = self.queries
        boxes_list = []
        class_ids_list = []
        confidences_list = []
        for i in range(len(batch["image_path"])):
            rgb_array = batch["rgb_image_array"][i]
            def callback(image_slice: np.ndarray) -> sv.Detections:
                # The slicer passes a numpy array. Convert to PIL for the transformer processor.
                # Note: PreProcessorStep already converted to RGB, so we don't need cvtColor here.
                image_pil = Image.fromarray(image_slice)

                inputs = self.processor(text=[queries], images=image_pil, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)

                target_sizes = torch.Tensor([image_pil.size[::-1]]).to(self.device)
                post_kwargs = {
                    "outputs": outputs,
                    "target_sizes": target_sizes,
                    "threshold": self.threshold,
                    "text_labels": [queries],
                }
                if self.text_threshold is not None:
                    post_kwargs["text_threshold"] = self.text_threshold
                results = self.processor.post_process_grounded_object_detection(**post_kwargs)

                # Build sv.Detections directly from boxes + scores + class
                # indices. Bypasses supervision's `from_transformers` (which
                # assumes `labels` is a torch tensor -- Grounding DINO returns
                # a Python list of class strings or ints).
                result = results[0]
                xyxy = result["boxes"].detach().cpu().numpy()
                scores = result["scores"].detach().cpu().numpy()

                labels_raw = result["labels"]
                if hasattr(labels_raw, "detach"):
                    class_idx = labels_raw.detach().cpu().numpy().astype(int)
                else:
                    # Python list -- entries may be ints or class-name strings
                    # (Grounding DINO with text_labels returns the latter).
                    class_idx = np.array(
                        [
                            queries.index(l) if isinstance(l, str) and l in queries
                            else (int(l) if not isinstance(l, str) else 0)
                            for l in labels_raw
                        ],
                        dtype=int,
                    )

                detections = sv.Detections(
                    xyxy=xyxy,
                    confidence=scores,
                    class_id=class_idx,
                )
                slice_area = image_slice.shape[0] * image_slice.shape[1]
                detections = detections[detections.area < (slice_area * 0.5)]

                return detections

            # Initialize the Slicer.
            # NON_MAX_SUPPRESSION (with a tighter IoU than the merge default)
            # drops overlapping cross-slice detections rather than merging them,
            # which keeps duplicate-piers down with OWLv2.
            slicer = sv.InferenceSlicer(
                callback=callback,
                # 960x960 matches OWLv2-large's native input (no resize artefacts)
                # and gives ~120m of context per slice at 12.5cm/pixel, comfortably
                # containing whole piers in this AOI.
                slice_wh=(960, 960),
                overlap_wh=(480, 480),
                overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
                iou_threshold=0.30,
            )

            # Run inference on the full image (slicer handles the tiling loop)
            detections = slicer(rgb_array)

            # Drop merged boxes that are unrealistically large -- a real pier
            # rarely spans more than ~120m (≈960 pixels at 12.5cm/pixel).
            w = detections.xyxy[:, 2] - detections.xyxy[:, 0]
            h = detections.xyxy[:, 3] - detections.xyxy[:, 1]
            detections = detections[(w < 960 * 0.8) & (h < 960 * 0.8)]
            boxes_list.append(detections.xyxy.astype(np.float32).tolist())
            class_ids_list.append(detections.class_id.astype(int).tolist())
            confidences_list.append(detections.confidence.astype(np.float32).tolist())

        batch["boxes"] = boxes_list
        batch["class_ids"] = class_ids_list
        batch["box_confidence"] = confidences_list
        return batch

# COMMAND ----------

# This class reprojects pixel-space bounding boxes into the source CRS using the
# per-image affine transform that PreProcessorStep already attached to the batch.
# Emits matching WKT POLYGON strings so the customer can compare detections to
# their BNG ground truth via IoU. Mirrors the logic in the customer's
# "Bounding Box Coordinate Transformation and IoU" notebook.
# Reprojects each pixel-space bbox to source-CRS coordinates and emits a
# matching WKT polygon. Optionally also computes per-image cluster envelopes
# (`cluster_bbox_wkt`) by buffering each AABB by `BUFFER_METRES` and joining
# overlapping ones via union-find.
#
# The dissolve was previously its own `map_batches` actor, but that forced an
# Arrow boundary between two pure-Python steps. On DBR's pyarrow + Ray combo,
# Ray's tensor-extension `ArrowPythonObjectScalar.as_py(maps_as_pydicts=...)`
# fails on ragged list-of-list columns like `box_world`. Folding the dissolve
# in here avoids the boundary entirely.
class BoxGeometryStep:
    BUFFER_METRES = 10

    def __init__(self, queries: list, run_dissolve: bool = False):
        # `queries` is the parsed open-vocabulary prompt list shared by
        # every image in the run; we resolve detector class indices to
        # human-readable labels here.
        self.queries = list(queries)
        self.run_dissolve = run_dissolve

    @staticmethod
    def _pixel_to_world(box, affine):
        # Ray's pyarrow round-trip occasionally returns more than 6 values for
        # the per-row affine (tensor-extension wrapping). Take the leading 6
        # which are the GDAL-style (a, b, c, d, e, f).
        a, b, c, d, e, f = list(affine)[:6]
        x1, y1, x2, y2 = list(box)[:4]
        return [
            a * x1 + b * y1 + c,
            d * x1 + e * y1 + f,
            a * x2 + b * y2 + c,
            d * x2 + e * y2 + f,
        ]

    @staticmethod
    def _world_box_to_wkt(world_box):
        # world_box = [easting_topleft, northing_topleft, easting_botright, northing_botright]
        # For north-up imagery (negative e in the affine) northing_topleft > northing_botright.
        e1, n1, e2, n2 = world_box
        return (
            f"POLYGON(({e1} {n2}, {e2} {n2}, {e2} {n1}, {e1} {n1}, {e1} {n2}))"
        )

    @staticmethod
    def _aabb(box_world):
        # `box_world` is [easting_topleft, northing_topleft,
        #                 easting_botright, northing_botright].
        e_tl, n_tl, e_br, n_br = box_world
        return (
            min(e_tl, e_br),
            min(n_tl, n_br),
            max(e_tl, e_br),
            max(n_tl, n_br),
        )

    def _dissolve(self, world_boxes, affine):
        """Returns ``(cluster_wkts, cluster_pixels)``: world-space WKT
        polygons of each cluster envelope, and the same envelopes converted
        to pixel space using ``affine`` (so Pipeline 2 can consume them
        directly without redoing the world->pixel inversion)."""
        if not world_boxes:
            return [], []
        extents = [self._aabb(list(b)[:4]) for b in world_boxes]
        n = len(extents)
        buf = self.BUFFER_METRES

        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for a in range(n):
            ax1, ay1, ax2, ay2 = extents[a]
            ax1 -= buf; ay1 -= buf; ax2 += buf; ay2 += buf
            for b in range(a + 1, n):
                bx1, by1, bx2, by2 = extents[b]
                bx1 -= buf; by1 -= buf; bx2 += buf; by2 += buf
                if ax1 <= bx2 and bx1 <= ax2 and ay1 <= by2 and by1 <= ay2:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[ra] = rb

        groups = {}
        for k in range(n):
            groups.setdefault(find(k), []).append(k)

        # Affine for the world->pixel inversion below. `b` and `d` are zero
        # for axis-aligned (north-up) imagery; `e` is negative, so world
        # ymax (top) maps to the smaller pixel y and ymin (bottom) maps to
        # the larger pixel y.
        aff_a, _, aff_c, _, aff_e, aff_f = list(affine)[:6]

        cluster_wkts = []
        cluster_pixels = []
        for indices in groups.values():
            xs = [v for k in indices for v in (extents[k][0], extents[k][2])]
            ys = [v for k in indices for v in (extents[k][1], extents[k][3])]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            cluster_wkts.append(
                f"POLYGON(({xmin} {ymin}, {xmax} {ymin}, "
                f"{xmax} {ymax}, {xmin} {ymax}, {xmin} {ymin}))"
            )
            cluster_pixels.append([
                (xmin - aff_c) / aff_a,
                (ymax - aff_f) / aff_e,
                (xmax - aff_c) / aff_a,
                (ymin - aff_f) / aff_e,
            ])
        return cluster_wkts, cluster_pixels

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Pixel-space and cluster bboxes are emitted as 4 flat per-image
        # `array<double>` columns rather than a single `array<array<double>>`,
        # because Ray's `materialize()` round-trips columns through Arrow and
        # the `ArrowPythonObjectScalar.as_py(maps_as_pydicts=...)` path blows
        # up on nested-list columns. The explode actors zip the 4 columns
        # back into a single 4-element list per detection.
        queries = self.queries
        wkt_list, class_label_list = [], []
        box_xmin, box_ymin, box_xmax, box_ymax = [], [], [], []
        cluster_wkt_list = []
        cluster_xmin, cluster_ymin, cluster_xmax, cluster_ymax = [], [], [], []
        for i in range(len(batch["image_path"])):
            affine = list(batch["transform"][i])
            raw_boxes = batch["boxes"][i]
            raw_class_ids = batch["class_ids"][i]
            if raw_boxes is None or len(raw_boxes) == 0:
                wkt_list.append([])
                class_label_list.append([])
                box_xmin.append([]); box_ymin.append([])
                box_xmax.append([]); box_ymax.append([])
                cluster_wkt_list.append([])
                cluster_xmin.append([]); cluster_ymin.append([])
                cluster_xmax.append([]); cluster_ymax.append([])
                continue
            boxes_iter = [
                b.tolist() if hasattr(b, "tolist") else list(b) for b in raw_boxes
            ]
            world_boxes = [self._pixel_to_world(b, affine) for b in boxes_iter]
            wkt_list.append([self._world_box_to_wkt(wb) for wb in world_boxes])
            class_label_list.append([
                queries[int(c)] if 0 <= int(c) < len(queries) else "unknown"
                for c in raw_class_ids
            ])
            box_xmin.append([float(b[0]) for b in boxes_iter])
            box_ymin.append([float(b[1]) for b in boxes_iter])
            box_xmax.append([float(b[2]) for b in boxes_iter])
            box_ymax.append([float(b[3]) for b in boxes_iter])

            if self.run_dissolve:
                wkts, pixels = self._dissolve(world_boxes, affine)
                cluster_wkt_list.append(wkts)
                cluster_xmin.append([float(p[0]) for p in pixels])
                cluster_ymin.append([float(p[1]) for p in pixels])
                cluster_xmax.append([float(p[2]) for p in pixels])
                cluster_ymax.append([float(p[3]) for p in pixels])
            else:
                cluster_wkt_list.append([])
                cluster_xmin.append([]); cluster_ymin.append([])
                cluster_xmax.append([]); cluster_ymax.append([])

        batch["box_wkt"] = wkt_list
        batch["box_class"] = class_label_list
        batch["box_pixel_xmin"] = box_xmin
        batch["box_pixel_ymin"] = box_ymin
        batch["box_pixel_xmax"] = box_xmax
        batch["box_pixel_ymax"] = box_ymax
        if self.run_dissolve:
            batch["cluster_bbox_wkt"] = cluster_wkt_list
            batch["cluster_pixel_xmin"] = cluster_xmin
            batch["cluster_pixel_ymin"] = cluster_ymin
            batch["cluster_pixel_xmax"] = cluster_xmax
            batch["cluster_pixel_ymax"] = cluster_ymax
        return batch

# COMMAND ----------

# Per-image -> per-detection explode actors. Replace the Spark posexplode +
# arrays_zip cells that previously sat between Ray write and the final
# per-detection tables. Running the explode in Ray means Delta tables come
# out per-detection straight from the pipeline, leaving Spark with only the
# small WKT->geometry lift.
#
# `map_batches` allows the output batch to have a different row count from
# the input, which is what makes the explode possible inside Ray.

def _opt(batch, key, i):
    """Defensive lookup: returns ``[]`` if ``batch[key][i]`` is missing or
    ``None``. Used by the explode actors which can run on batches whose
    upstream produced empty/zero-detection rows."""
    if key not in batch:
        return []
    val = batch[key][i]
    return [] if val is None else val


class ExplodeBoxesStep:
    """Per-image rows -> one row per bounding box. Emits both the world-space
    `box_wkt` and the pixel-space `box_pixel` so Pipeline 2 can prompt SAM
    directly without redoing the world->pixel inversion. The pixel coords
    are zipped from 4 flat per-image arrays to dodge the pyarrow nested-list
    serialisation bug at the materialize() boundary."""

    def __init__(self):
        pass

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        out = {
            "image_path": [],
            "crs": [],
            "transform": [],
            "box_idx": [],
            "box_wkt": [],
            "box_pixel": [],
            "box_class": [],
            "box_confidence": [],
        }
        for i in range(len(batch["image_path"])):
            wkts = _opt(batch, "box_wkt", i)
            xmins = _opt(batch, "box_pixel_xmin", i)
            ymins = _opt(batch, "box_pixel_ymin", i)
            xmaxs = _opt(batch, "box_pixel_xmax", i)
            ymaxs = _opt(batch, "box_pixel_ymax", i)
            classes = _opt(batch, "box_class", i)
            confs = _opt(batch, "box_confidence", i)
            for k in range(len(wkts)):
                out["image_path"].append(batch["image_path"][i])
                out["crs"].append(batch["crs"][i])
                out["transform"].append(list(batch["transform"][i]))
                out["box_idx"].append(k)
                out["box_wkt"].append(wkts[k])
                out["box_pixel"].append([
                    float(xmins[k]), float(ymins[k]),
                    float(xmaxs[k]), float(ymaxs[k]),
                ])
                out["box_class"].append(classes[k])
                out["box_confidence"].append(float(confs[k]))
        return out

# COMMAND ----------

class ExplodeClustersStep:
    """Per-image rows -> one row per cluster envelope. Emits both the
    world-space `cluster_bbox_wkt` and the pixel-space `cluster_pixel`
    pre-computed by `BoxGeometryStep`. Same flat-array trick as the boxes
    explode."""

    def __init__(self):
        pass

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        out = {
            "image_path": [],
            "cluster_id": [],
            "cluster_bbox_wkt": [],
            "cluster_pixel": [],
        }
        for i in range(len(batch["image_path"])):
            wkts = _opt(batch, "cluster_bbox_wkt", i)
            xmins = _opt(batch, "cluster_pixel_xmin", i)
            ymins = _opt(batch, "cluster_pixel_ymin", i)
            xmaxs = _opt(batch, "cluster_pixel_xmax", i)
            ymaxs = _opt(batch, "cluster_pixel_ymax", i)
            for k in range(len(wkts)):
                out["image_path"].append(batch["image_path"][i])
                out["cluster_id"].append(k)
                out["cluster_bbox_wkt"].append(wkts[k])
                out["cluster_pixel"].append([
                    float(xmins[k]), float(ymins[k]),
                    float(xmaxs[k]), float(ymaxs[k]),
                ])
        return out

# COMMAND ----------

class ExplodeSegmentsStep:
    """Per-image rows -> one row per segmentation polygon."""

    def __init__(self):
        pass

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        out = {
            "image_path": [],
            "crs": [],
            "transform": [],
            "seg_idx": [],
            "seg_wkt": [],
        }
        for i in range(len(batch["image_path"])):
            wkts = batch["mask_wkt"][i] if batch["mask_wkt"][i] is not None else []
            for k in range(len(wkts)):
                out["image_path"].append(batch["image_path"][i])
                out["crs"].append(batch["crs"][i])
                out["transform"].append(list(batch["transform"][i]))
                out["seg_idx"].append(k)
                out["seg_wkt"].append(wkts[k])
        return out

# COMMAND ----------

# Common scaffolding for the segmentation+polygonisation actors. Subclasses
# load a specific SAM backend in `__init__` and implement `_predict_masks`,
# which returns an `(N, H, W)` boolean/integer mask array for a batch of
# pixel-space box prompts.
#
# The merged `__call__` does both inference and polygonisation in the same
# Ray actor: pyarrow's `ArrowPythonObjectScalar.as_py(maps_as_pydicts=...)`
# blows up when image-shaped numpy arrays cross a `map_batches` boundary,
# so the encoded mask never leaves Python.
class _BaseSegmentAndPolygonizeStep:
    WORKING_TYPE = np.uint32
    TARGET_TYPE = np.uint16
    MAX_VAL = np.iinfo(TARGET_TYPE).max
    SUB_BATCH_SIZE = 50

    def _predict_masks(self, image_pil, boxes_np, hw):
        raise NotImplementedError

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        from rasterio.features import shapes as rio_shapes
        from rasterio.transform import Affine
        from shapely.geometry import shape as shapely_shape

        wkts_list = []
        for i in range(len(batch["image_path"])):
            rgb_array = batch["rgb_image_array"][i]
            original_np = batch["original_image_np"][i]
            raw_boxes = batch["boxes"][i]
            transform = list(batch["transform"][i])[:6]

            if raw_boxes is None or len(raw_boxes) == 0:
                boxes_np = np.empty((0, 4), dtype=np.float32)
            else:
                processed_boxes = [
                    b.tolist() if hasattr(b, "tolist") else b for b in raw_boxes
                ]
                boxes_np = np.array(processed_boxes, dtype=np.float32)
            boxes_np = np.atleast_2d(boxes_np)

            mask_overlay = np.zeros(original_np.shape[:2], dtype=self.WORKING_TYPE)
            if boxes_np.shape[0] > 0:
                image_pil = Image.fromarray(rgb_array)
                masks = self._predict_masks(image_pil, boxes_np, original_np.shape[:2])
                for j in range(masks.shape[0]):
                    mask_overlay[masks[j] > 0] = j + 1

            if mask_overlay.max() > self.MAX_VAL:
                mask_overlay = np.clip(mask_overlay, 0, self.MAX_VAL)
            encoded_mask = mask_overlay.astype(self.TARGET_TYPE)

            polys = []
            object_ids = np.unique(encoded_mask[encoded_mask > 0])
            if len(object_ids) > 0:
                affine = Affine(*transform)
                for obj_id in object_ids:
                    single_obj = (encoded_mask == obj_id).astype(np.uint8)
                    for geom, val in rio_shapes(single_obj, transform=affine):
                        if val == 1:
                            polys.append(shapely_shape(geom).wkt)
            wkts_list.append(polys)

        batch["mask_wkt"] = wkts_list
        return batch

# COMMAND ----------

class SegmentAndPolygonizeStep(_BaseSegmentAndPolygonizeStep):
    """SAM2 backend (`samgeo.samgeo2.SamGeo2` with `sam2-hiera-small`)."""

    SAM_MODEL = "sam2-hiera-small"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sam = SamGeo2(model_id=self.SAM_MODEL, device=self.device, automatic=False)
        print("SegmentAndPolygonizeStep initialized on device:", self.device)

    def _predict_masks(self, image_pil, boxes_np, hw):
        self.sam.set_image(image_pil)
        chunks = []
        for k in range(0, boxes_np.shape[0], self.SUB_BATCH_SIZE):
            # samgeo2.predict only forwards `boxes` to SAM2 if it's a Python
            # list -- numpy arrays are silently treated as None. `.tolist()`
            # ensures the prompt actually lands.
            sub_boxes = boxes_np[k:k + self.SUB_BATCH_SIZE].tolist()
            masks, _, _ = self.sam.predict(
                boxes=sub_boxes,
                multimask_output=False,
                return_results=True,
            )
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)
            chunks.append(masks)
        return np.concatenate(chunks, axis=0)

# COMMAND ----------

class SegmentAndPolygonizeStepSAM3(_BaseSegmentAndPolygonizeStep):
    """SAM3 backend (`samgeo.samgeo3.SamGeo3` with the meta backend in
    interactive mode). Selected by `SEGMENTER_VERSION = "samgeo3"`."""

    def __init__(self):
        from samgeo.samgeo3 import SamGeo3
        import os
        import sam3

        # samgeo3 defaults to looking in `samgeo/assets/` for the BPE vocab,
        # but the file is shipped with the `sam3` package itself at
        # `sam3/assets/bpe_simple_vocab_16e6.txt.gz`. Resolve sam3's install
        # location and point samgeo3 there.
        bpe_path = os.path.join(
            os.path.dirname(sam3.__file__),
            "assets",
            "bpe_simple_vocab_16e6.txt.gz",
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sam = SamGeo3(
            backend="meta",
            device=self.device,
            enable_inst_interactivity=True,
            bpe_path=bpe_path,
        )
        print(f"SegmentAndPolygonizeStepSAM3 initialized on device: {self.device}")

    def _predict_masks(self, image_pil, boxes_np, hw):
        H, W = hw
        chunks = []
        # SAM3 expects bfloat16 inputs while keeping some Linear weights in
        # float32, raising "mat1 and mat2 must have the same dtype" without
        # the autocast wrapper. See facebookresearch/sam3#507.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.sam.set_image(image_pil)
            for k in range(0, boxes_np.shape[0], self.SUB_BATCH_SIZE):
                sub_boxes = boxes_np[k:k + self.SUB_BATCH_SIZE].tolist()
                # `generate_masks_by_boxes_inst` returns None; masks land on
                # `self.sam.masks`. Default `box_crs=None` means pixel coords.
                self.sam.generate_masks_by_boxes_inst(sub_boxes)
                masks = self.sam.masks
                # bfloat16 -> float32 -> numpy (numpy can't read bf16).
                if hasattr(masks, "detach"):
                    masks = masks.detach().float().cpu().numpy()
                masks = np.asarray(masks)
                if masks.ndim == 4 and masks.shape[1] == 1:
                    masks = masks.squeeze(1)
                if masks.ndim == 2:
                    masks = masks[None, ...]
                assert masks.ndim == 3 and masks.shape[-2:] == (H, W), (
                    f"Unexpected SAM3 masks shape {masks.shape}; "
                    f"expected (..., {H}, {W})."
                )
                chunks.append(masks)
        return np.concatenate(chunks, axis=0)

# COMMAND ----------

# Detection prompt is a run-wide constant -- the actor closes over it
# instead of carrying it as a per-row column.
queries = [q.strip() for q in text_prompt.split(",") if q.strip()]

data = spark.createDataFrame([(image_path,)], ["image_path"])
display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC This could use some tweaking to maximise resource use.

# COMMAND ----------

ds = ray.data.from_spark(data)

# Step 1: PreProcessorStep (CPU-bound)
# This is a lightweight, CPU-only task. We can create a larger pool of actors
# to ensure it doesn't become a bottleneck for the GPU stages.
ds = ds.map_batches(
    PreProcessorStep,
    concurrency=(1, 3 * (1 + max_worker_nodes)),
    batch_size=2,
    num_cpus=1,
)

# Step 2: BBoxPredictorStep (High VRAM GPU task)
# We request 0.5 of a GPU for each actor. With 3 GPUs, Ray can place 3 actors
# across the cluster, one on each GPU, leaving room for the next step.
ds = ds.map_batches(
    BBoxPredictorStep,
    fn_constructor_kwargs={"queries": queries},
    concurrency=(1, 1 + max_worker_nodes),
    batch_size=4,
    num_gpus=0.75, # Request 50% of a GPU (~20GB on an A100/40GB)
)

# Step 3: BoxGeometryStep (CPU-only, lightweight)
# Reproject pixel boxes to source-CRS coordinates using the per-image affine
# and emit WKT polygons for downstream IoU evaluation. When `RUN_DISSOLVE` is
# on, this same actor also computes per-image cluster envelopes
# (`cluster_bbox_wkt`) -- folded into one map_batches to avoid the Arrow
# boundary that breaks Ray's tensor-extension scalar conversion on this
# pyarrow.
ds = ds.map_batches(
    BoxGeometryStep,
    fn_constructor_kwargs={"queries": queries, "run_dissolve": RUN_DISSOLVE},
    concurrency=(1, 3 * (1 + max_worker_nodes)),
    batch_size=8,
    num_cpus=1,
)

# Segmentation runs as a separate Ray pipeline further down, sourced from
# whichever per-detection table Pipeline 1 produced (clusters_table when
# RUN_DISSOLVE=true, boxes_table when false).

# `boxes` is a per-image list-of-4-tuples (array<array<double>>) which Ray
# can't safely round-trip across the materialize() boundary -- pyarrow's
# `ArrowPythonObjectScalar.as_py(maps_as_pydicts=...)` blows up on nested
# lists. BoxGeometryStep already derived the per-image flat
# `box_pixel_xmin/ymin/xmax/ymax` arrays from `boxes`, so we drop the
# nested column here.
ds = ds.drop_columns([
    "rgb_image_array", "original_image_np", "boxes", "class_ids",
])

# COMMAND ----------

# MAGIC %md
# MAGIC Ray needs a temp folder to write datasets to Delta. 

# COMMAND ----------

temp_dir = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/outputs"
dbutils.fs.mkdirs(temp_dir)
os.environ["RAY_UC_VOLUMES_FUSE_TEMP_DIR"] = temp_dir

# COMMAND ----------

# MAGIC %md
# MAGIC Set a table name

# COMMAND ----------

table_name = "inference_results"
tref = f"{CATALOG}.{SCHEMA}.{table_name}"

boxes_table = f"{tref}_boxes"
clusters_table = f"{tref}_clusters"

# Materialise once so the box-explode and cluster-explode forks below don't
# each re-run the upstream actors. After this, `ds` is a MaterializedDataset
# in the object store and either branch reads from it cheaply.
ds = ds.materialize()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Per-box geometry table for IoU evaluation
# MAGIC
# MAGIC `ExplodeBoxesStep` flattens the per-image detection arrays into one
# MAGIC row per bounding box inside Ray. Spark's only role afterwards is the
# MAGIC tiny WKT-to-`geometry` lift -- no posexplode, no `arrays_zip`, no
# MAGIC tensor-extension struct unpacking.

# COMMAND ----------

ds_boxes = ds.map_batches(
    ExplodeBoxesStep,
    concurrency=(1, 3 * (1 + max_worker_nodes)),
    batch_size=8,
    num_cpus=1,
)

boxes_stage = f"{boxes_table}_wkt"
spark.sql(f"DROP TABLE IF EXISTS {boxes_stage}")
ds_boxes.write_databricks_table(boxes_stage, mode='overwrite')

# Single Spark transform: lift `box_wkt` to a Databricks `geometry` column.
spark.sql(f"DROP TABLE IF EXISTS {boxes_table}")
spark.sql(f"""
    CREATE TABLE {boxes_table} AS
    SELECT *, st_geomfromtext(box_wkt, {SRID}) AS geometry
    FROM {boxes_stage}
""")
spark.sql(f"DROP TABLE {boxes_stage}")
display(spark.table(boxes_table))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Per-cluster geometry table
# MAGIC
# MAGIC Skipped when `RUN_DISSOLVE = False`. `ExplodeClustersStep` flattens
# MAGIC the per-image `cluster_bbox_wkt` arrays into one row per cluster
# MAGIC inside Ray; Spark only adds the `cluster_bbox` geometry column.

# COMMAND ----------

if RUN_DISSOLVE:
    ds_clusters = ds.map_batches(
        ExplodeClustersStep,
        concurrency=(1, 3 * (1 + max_worker_nodes)),
        batch_size=8,
        num_cpus=1,
    )

    clusters_stage = f"{clusters_table}_wkt"
    spark.sql(f"DROP TABLE IF EXISTS {clusters_stage}")
    ds_clusters.write_databricks_table(clusters_stage, mode='overwrite')

    spark.sql(f"DROP TABLE IF EXISTS {clusters_table}")
    spark.sql(f"""
        CREATE TABLE {clusters_table} AS
        SELECT *, st_geomfromtext(cluster_bbox_wkt, {SRID}) AS cluster_bbox
        FROM {clusters_stage}
    """)
    spark.sql(f"DROP TABLE {clusters_stage}")
    display(spark.table(clusters_table))
else:
    print(
        f"RUN_DISSOLVE=False -- skipping cluster table. Pipeline 2 and the "
        f"evaluation will use raw per-box geometries from {boxes_table}."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline 2: Segmentation on dissolved cluster bboxes
# MAGIC
# MAGIC The dissolved `cluster_bbox` envelopes are clean inputs for SAM2 -- one
# MAGIC bbox per physical pier rather than the original swarm of overlapping
# MAGIC detections. Pipeline 1 already persisted each detection's pixel-space
# MAGIC bbox (`box_pixel` / `cluster_pixel`), so this pipeline just re-loads
# MAGIC the image, hands the pixel boxes to SAM, and polygonises the masks
# MAGIC back to world-space WKT.
# MAGIC
# MAGIC Skipped entirely when `RUN_SEGMENTATION = False`.

# COMMAND ----------

if RUN_SEGMENTATION and SEGMENTER_VERSION == "samgeo3":
    # SAM3's weights are gated on Hugging Face. Set HF_TOKEN here so it
    # propagates to every Ray actor for model download. Stored in the
    # `stuart` Databricks secret scope under key `hf_token`.
    import os
    os.environ["HF_TOKEN"] = dbutils.secrets.get(scope="stuart", key="hf_token")
    print("HF_TOKEN set from Databricks secret stuart/hf_token.")

# COMMAND ----------

if RUN_SEGMENTATION:
    # Build the per-image segmentation input: one row per image. We pass each
    # bbox coord as its own flat array<double> column rather than a nested
    # array<array<double>>, because Ray's from_spark trips over the latter
    # (pyarrow ArrowPythonObjectScalar incompatibility with `maps_as_pydicts`).
    #
    # Source table flips with `RUN_DISSOLVE`: cluster envelopes when on,
    # raw per-detection geometries when off. Both tables now carry a
    # pixel-space bbox column (`box_pixel` / `cluster_pixel`) pre-computed
    # in Pipeline 1, so no world->pixel inversion is needed here.
    _bbox_source = clusters_table if RUN_DISSOLVE else boxes_table
    _bbox_pixel_col = "cluster_pixel" if RUN_DISSOLVE else "box_pixel"
    seg_input_df = spark.sql(f"""
        SELECT
            image_path,
            collect_list({_bbox_pixel_col}[0]) AS xmins,
            collect_list({_bbox_pixel_col}[1]) AS ymins,
            collect_list({_bbox_pixel_col}[2]) AS xmaxs,
            collect_list({_bbox_pixel_col}[3]) AS ymaxs
        FROM {_bbox_source}
        GROUP BY image_path
    """)
    display(seg_input_df)

# COMMAND ----------

# Zips the per-image pixel-space xmins/ymins/xmaxs/ymaxs columns (already
# pre-computed by Pipeline 1 and carried through `seg_input_df` as four
# flat array<double> columns) into the `batch["boxes"]` list-of-4-tuples
# format the segmentation actors expect. No coordinate transforms here:
# the world->pixel inversion lives once, in `BoxGeometryStep`.
class PixelBoxAssemblerStep:
    def __init__(self):
        pass

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        boxes_per_image = []
        for i in range(len(batch["image_path"])):
            xmins = list(batch["xmins"][i])
            ymins = list(batch["ymins"][i])
            xmaxs = list(batch["xmaxs"][i])
            ymaxs = list(batch["ymaxs"][i])
            boxes_per_image.append(
                [list(b) for b in zip(xmins, ymins, xmaxs, ymaxs)]
            )

        batch["boxes"] = boxes_per_image
        return batch

# COMMAND ----------

if RUN_SEGMENTATION:
    seg_ds = ray.data.from_spark(seg_input_df)

    # Re-open the GeoTIFF / ECW for each image to recover rgb_image_array,
    # original_image_np, crs, transform.
    seg_ds = seg_ds.map_batches(
        PreProcessorStep,
        concurrency=(1, 3 * (1 + max_worker_nodes)),
        batch_size=2,
        num_cpus=1,
    )

    # Zip the per-image pixel-space xmins/ymins/xmaxs/ymaxs into the
    # `batch["boxes"]` list-of-4-tuples format SAM expects. No coordinate
    # conversion -- pixel boxes were pre-computed in Pipeline 1.
    seg_ds = seg_ds.map_batches(
        PixelBoxAssemblerStep,
        concurrency=(1, 3 * (1 + max_worker_nodes)),
        batch_size=8,
        num_cpus=1,
    )

    # Segmentation + in-step polygonisation. Merged into a single map_batches
    # actor so the bytes-typed `mask` column never has to be passed between
    # Ray actors. Backend selected by `SEGMENTER_VERSION` near the top of the
    # notebook -- `samgeo2` (SAM2) or `samgeo3` (SAM3, transformers backend).
    _seg_step = {
        "samgeo2": SegmentAndPolygonizeStep,
        "samgeo3": SegmentAndPolygonizeStepSAM3,
    }[SEGMENTER_VERSION]

    # samgeo3 downloads gated SAM3 weights from Hugging Face inside the
    # actor's __init__, and driver-side os.environ doesn't propagate to
    # actors on other nodes. Forward HF_TOKEN through `runtime_env`, which
    # `map_batches` accepts via its **ray_remote_args collector and applies
    # to each actor's process before __init__ runs.
    _seg_extra = {}
    if SEGMENTER_VERSION == "samgeo3":
        _seg_extra = {
            "runtime_env": {"env_vars": {"HF_TOKEN": os.environ["HF_TOKEN"]}}
        }

    seg_ds = seg_ds.map_batches(
        _seg_step,
        concurrency=(1, 1 + max_worker_nodes),
        batch_size=8,
        num_gpus=0.4,
        **_seg_extra,
    )

    seg_ds = seg_ds.drop_columns(
        ["rgb_image_array", "original_image_np", "boxes",
         "xmins", "ymins", "xmaxs", "ymaxs"]
    )

    # Per-image -> per-segment explode in Ray. Same pattern as Pipeline 1:
    # the per-detection write happens straight from Ray, leaving Spark only
    # the WKT->geometry lift.
    seg_ds = seg_ds.map_batches(
        ExplodeSegmentsStep,
        concurrency=(1, 3 * (1 + max_worker_nodes)),
        batch_size=8,
        num_cpus=1,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Per-segment geometry table
# MAGIC
# MAGIC `ExplodeSegmentsStep` flattens the per-image `mask_wkt` arrays in Ray;
# MAGIC Spark only lifts `seg_wkt` to a Databricks `geometry` column.

# COMMAND ----------

if RUN_SEGMENTATION:
    segments_table = f"{tref}_segments_{SEGMENTER_VERSION}"
    segments_stage = f"{segments_table}_wkt"

    spark.sql(f"DROP TABLE IF EXISTS {segments_stage}")
    seg_ds.write_databricks_table(segments_stage, mode="overwrite")

    spark.sql(f"DROP TABLE IF EXISTS {segments_table}")
    spark.sql(f"""
        CREATE TABLE {segments_table} AS
        SELECT *, st_geomfromtext(seg_wkt, {SRID}) AS geometry
        FROM {segments_stage}
    """)
    spark.sql(f"DROP TABLE {segments_stage}")
    display(spark.table(segments_table))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation against ground truth
# MAGIC
# MAGIC Compares detection geometries to the ground-truth polygons in
# MAGIC `${CATALOG}.${SCHEMA}.ground_truth` (populated by the
# MAGIC `load_ground_truth` notebook).
# MAGIC
# MAGIC The same evaluator runs after each Ray pipeline:
# MAGIC
# MAGIC - **Bounding-box detections** -- the detector emits axis-aligned
# MAGIC   rectangles, so we take `st_envelope` of each ground-truth polygon
# MAGIC   first. Apples-to-apples bbox comparison.
# MAGIC - **Segmentation polygons** -- detection masks already trace the
# MAGIC   structure's actual shape, so we compare the raw GT polygon directly.
# MAGIC
# MAGIC Reports:
# MAGIC
# MAGIC - **Image-level IoU** -- the dissolve-and-intersect approach from the
# MAGIC   customer's reference notebook (single number).
# MAGIC - **Per-detection matching** -- for each detection, the IoU of its
# MAGIC   best ground-truth match; same for each ground-truth row. From those,
# MAGIC   precision, recall and F1 are computed at `IOU_MATCH_THRESHOLD`.

# COMMAND ----------

GROUND_TRUTH_TABLE = f"{CATALOG}.{SCHEMA}.ground_truth"

# COMMAND ----------

def evaluate_against_ground_truth(
    det_table: str,
    det_id_col: str,
    det_geom_col: str,
    *,
    envelope_gt: bool,
    label: str,
    iou_threshold: float = IOU_MATCH_THRESHOLD,
):
    """
    Score detections in ``det_table`` against ``GROUND_TRUTH_TABLE``.

    ``envelope_gt`` should be True for axis-aligned bbox detections (where
    the ground-truth polygons get reduced to their MBRs first) and False
    for segmentation polygons (compared directly).

    ``label`` is used in temp-view names and the printed summary header so
    multiple invocations don't collide.
    """
    if not spark.catalog.tableExists(GROUND_TRUTH_TABLE):
        print(f"Ground truth table {GROUND_TRUTH_TABLE} not found.")
        print("Run the `load_ground_truth` notebook (in src/) before evaluating.")
        return None

    safe = label.replace("-", "_").replace(" ", "_").lower()
    gt_view = f"gt_{safe}"
    pair_view = f"pair_iou_{safe}"
    det_match_view = f"det_match_{safe}"
    gt_match_view = f"gt_match_{safe}"

    gt_geom_expr = "st_envelope(geometry)" if envelope_gt else "geometry"
    spark.sql(f"""
        CREATE OR REPLACE TEMP VIEW {gt_view} AS
        SELECT gt_id, {gt_geom_expr} AS geometry
        FROM {GROUND_TRUTH_TABLE}
    """)

    # Image-level IoU -- dissolve both sets, then compute IoU of the unions.
    image_iou = spark.sql(f"""
        WITH gt AS (SELECT st_union_agg(geometry) AS geom FROM {gt_view}),
             pr AS (SELECT st_union_agg({det_geom_col}) AS geom FROM {det_table}),
             areas AS (
                 SELECT st_area(g.geom) AS area_gt,
                        st_area(p.geom) AS area_pr,
                        st_area(st_intersection(g.geom, p.geom)) AS area_intersect
                 FROM gt g, pr p
             )
        SELECT area_gt, area_pr, area_intersect,
               area_gt + area_pr - area_intersect AS area_union,
               CASE WHEN (area_gt + area_pr - area_intersect) > 0
                    THEN area_intersect / (area_gt + area_pr - area_intersect)
                    ELSE 0 END AS iou_image_level
        FROM areas
    """)
    display(image_iou)

    # Pairwise IoU across every (detection, ground-truth) pair that intersects.
    spark.sql(f"""
        CREATE OR REPLACE TEMP VIEW {pair_view} AS
        SELECT
            gt.gt_id,
            det.image_path,
            det.{det_id_col} AS det_id,
            st_area(st_intersection(gt.geometry, det.{det_geom_col})) /
                NULLIF(
                    st_area(gt.geometry) + st_area(det.{det_geom_col})
                        - st_area(st_intersection(gt.geometry, det.{det_geom_col})),
                    0
                ) AS iou
        FROM {gt_view} gt
        CROSS JOIN {det_table} det
        WHERE st_intersects(gt.geometry, det.{det_geom_col})
    """)

    # Best matching IoU per detection (precision side) and per ground truth (recall side).
    spark.sql(f"""
        CREATE OR REPLACE TEMP VIEW {det_match_view} AS
        SELECT det.image_path,
               det.{det_id_col} AS det_id,
               COALESCE(MAX(p.iou), 0.0) AS best_iou
        FROM {det_table} det
        LEFT JOIN {pair_view} p
            ON det.image_path = p.image_path
            AND det.{det_id_col} = p.det_id
        GROUP BY det.image_path, det.{det_id_col}
    """)
    spark.sql(f"""
        CREATE OR REPLACE TEMP VIEW {gt_match_view} AS
        SELECT gt.gt_id,
               COALESCE(MAX(p.iou), 0.0) AS best_iou
        FROM {gt_view} gt
        LEFT JOIN {pair_view} p USING (gt_id)
        GROUP BY gt.gt_id
    """)

    det_count, det_tp = spark.sql(
        f"SELECT COUNT(*), SUM(CASE WHEN best_iou >= {iou_threshold} THEN 1 ELSE 0 END) FROM {det_match_view}"
    ).first()
    gt_count, gt_tp = spark.sql(
        f"SELECT COUNT(*), SUM(CASE WHEN best_iou >= {iou_threshold} THEN 1 ELSE 0 END) FROM {gt_match_view}"
    ).first()

    det_count, gt_count = int(det_count or 0), int(gt_count or 0)
    det_tp, gt_tp = int(det_tp or 0), int(gt_tp or 0)

    precision = det_tp / det_count if det_count else 0.0
    recall    = gt_tp  / gt_count  if gt_count  else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"--- Evaluation: {label} ---")
    print(f"Detection table      : {det_table}")
    print(f"GT geometry          : {'envelope' if envelope_gt else 'raw polygon'}")
    print(f"IoU match threshold  : {iou_threshold}")
    print(f"Ground-truth count   : {gt_count}")
    print(f"Detection count      : {det_count}")
    print(f"Detections matched   : {det_tp}    -> precision = {precision:.3f}")
    print(f"Ground-truth matched : {gt_tp}    -> recall    = {recall:.3f}")
    print(f"F1                   : {f1:.3f}")

    return {
        "label": label,
        "det_table": det_table,
        "envelope_gt": envelope_gt,
        "iou_threshold": iou_threshold,
        "gt_count": gt_count,
        "det_count": det_count,
        "det_tp": det_tp,
        "gt_tp": gt_tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate bounding-box detections

# COMMAND ----------

# Source flips with `RUN_DISSOLVE`: cluster envelopes when on, raw
# per-detection geometries when off. Both share `image_path` but use
# different per-row keys and a different geometry column.
if RUN_DISSOLVE:
    bbox_metrics = evaluate_against_ground_truth(
        det_table=clusters_table,
        det_id_col="cluster_id",
        det_geom_col="cluster_bbox",
        envelope_gt=True,
        label="clusters",
    )
else:
    bbox_metrics = evaluate_against_ground_truth(
        det_table=boxes_table,
        det_id_col="box_idx",
        det_geom_col="geometry",
        envelope_gt=True,
        label="boxes",
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate segmentation polygons
# MAGIC
# MAGIC Skipped when `RUN_SEGMENTATION = False`. Segmentation polygons trace
# MAGIC the structure shape directly, so the ground-truth polygons are used
# MAGIC as-is (no envelope step).

# COMMAND ----------

if RUN_SEGMENTATION:
    seg_metrics = evaluate_against_ground_truth(
        det_table=segments_table,
        det_id_col="seg_idx",
        det_geom_col="geometry",
        envelope_gt=False,
        label=f"segments_{SEGMENTER_VERSION}",
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Widget definitions
# MAGIC
# MAGIC Everything below `dbutils.notebook.exit()` is skipped on a normal "Run
# MAGIC All". Run the next cell once to declare/refresh the notebook's widget
# MAGIC bar; on jobs the bar is populated from `base_parameters` in
# MAGIC `resources/ray_inference.job.yml` and this cell never needs to run.
# MAGIC
# MAGIC The widgets exposed here are the run-time knobs the read cell at the
# MAGIC top of the notebook consumes:
# MAGIC
# MAGIC - `text_prompt` -- comma-separated open-vocabulary detection prompt.
# MAGIC - `BBOX_MODEL` -- one of `owlv2`, `grounding_dino`, `omdet`. OWLv2 is
# MAGIC   the strongest stock choice on this aerial AOI; grounding_dino has
# MAGIC   broader text matching but more false positives; omdet is fastest
# MAGIC   but weakest.
# MAGIC - `RUN_SEGMENTATION` -- `true` to run SAM2/SAM3, `false` to stop after
# MAGIC   the bbox + IoU-against-ground-truth stage.
# MAGIC - `SEGMENTER_VERSION` -- `samgeo2` (SamGeo2 / sam2-hiera-small) or
# MAGIC   `samgeo3` (SamGeo3 / facebook/sam3, transformers backend).
# MAGIC - `RUN_DISSOLVE` -- `true` to buffer-and-union overlapping bbox
# MAGIC   detections into cluster envelopes before the segmenter sees them
# MAGIC   (essential for Thames; muddies Norfolk's small/close structures).
# MAGIC - `CATALOG`, `SCHEMA`, `VOLUME` -- Unity Catalog locations for input
# MAGIC   imagery and output tables.
# MAGIC - `image_path` -- full Volumes path to the source image.
# MAGIC - `SRID` -- EPSG code for the source imagery's CRS (default 27700).
# MAGIC - `IOU_MATCH_THRESHOLD` -- min IoU above which a detection counts as
# MAGIC   a true positive in the per-detection precision/recall/F1 summary.

# COMMAND ----------

dbutils.notebook.exit("done")

# COMMAND ----------

# This cell is dormant on a "Run All" because the cell above exits the
# notebook. Run it manually once (or via Run > Run cell) to populate the
# widget bar. Repeated runs harmlessly re-declare each widget at its
# default value, then `removeAll()` strips any widgets from earlier
# notebook revisions.
dbutils.widgets.removeAll()

dbutils.widgets.text("text_prompt", "pier", "Detection prompt (comma-separated)")
dbutils.widgets.dropdown(
    "BBOX_MODEL", "owlv2",
    ["owlv2", "grounding_dino", "omdet"],
    "Bounding-box detector",
)
dbutils.widgets.dropdown(
    "RUN_SEGMENTATION", "true",
    ["true", "false"],
    "Run segmentation",
)
dbutils.widgets.dropdown(
    "SEGMENTER_VERSION", "samgeo2",
    ["samgeo2", "samgeo3"],
    "Segmentation backend",
)
dbutils.widgets.dropdown(
    "RUN_DISSOLVE", "true",
    ["true", "false"],
    "Dissolve overlapping detections",
)
dbutils.widgets.text("CATALOG", "stuart",  "Catalog")
dbutils.widgets.text("SCHEMA",  "tce",     "Schema")
dbutils.widgets.text("VOLUME",  "imagery", "Volume")
dbutils.widgets.text(
    "image_path",
    "/Volumes/stuart/tce/imagery/NorfolkAOI/Ortho_RGBN_P00116896_20240623_20240623_20cm_res.ecw",
    "Source image path",
)
dbutils.widgets.text("SRID", "27700", "Source CRS (EPSG)")
dbutils.widgets.text("IOU_MATCH_THRESHOLD", "0.5", "IoU threshold for TP")

