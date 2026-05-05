# Databricks notebook source
# MAGIC %pip install uv

# COMMAND ----------

# MAGIC %sh
# MAGIC uv pip install \
# MAGIC   --no-binary rasterio \
# MAGIC   --excludes ../excludes.txt \
# MAGIC   -r ../requirements.txt

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os

import cv2
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

from pyspark.sql import functions as F
from pyspark.databricks.sql import functions as DBF

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

    def __init__(self):
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

        print(f"BBoxPredictorStep initialized: model={self.model_name}, device={self.device}")

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Processes a batch of images and text prompts to find bounding boxes.
        - Converts NumPy image arrays to PIL Images.
        - Runs the configured open-vocabulary detector on each slice.
        - Adds the predicted pixel-space boxes to the batch.
        """
        boxes_list = []
        class_ids_list = []
        confidences_list = []
        for i in range(len(batch["image_path"])):
            rgb_array = batch["rgb_image_array"][i]
            # `text_prompt` may be comma-separated; all three detectors accept a
            # list of class queries per image and return boxes from each.
            text_prompt = batch["text_prompt"][i]
            queries = [q.strip() for q in text_prompt.split(",") if q.strip()]
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
class BoxGeometryStep:
    def __init__(self):
        pass

    @staticmethod
    def _pixel_to_world(box, affine):
        # Defensive: Ray's pyarrow serialisation of the per-row affine occasionally
        # comes back longer than 6 (extension types / nested wrapping). Take the
        # leading 6 floats which are the GDAL-style (a, b, c, d, e, f).
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

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        world_list, wkt_list, class_label_list = [], [], []
        for i in range(len(batch["image_path"])):
            affine = list(batch["transform"][i])
            raw_boxes = batch["boxes"][i]
            raw_class_ids = batch["class_ids"][i]
            queries = [
                q.strip() for q in batch["text_prompt"][i].split(",") if q.strip()
            ]
            if raw_boxes is None or len(raw_boxes) == 0:
                world_list.append([])
                wkt_list.append([])
                class_label_list.append([])
                continue
            boxes_iter = [
                b.tolist() if hasattr(b, "tolist") else list(b) for b in raw_boxes
            ]
            world_boxes = [self._pixel_to_world(b, affine) for b in boxes_iter]
            world_list.append(world_boxes)
            wkt_list.append([self._world_box_to_wkt(wb) for wb in world_boxes])
            # Map class indices back to the human-readable query strings.
            class_label_list.append([
                queries[int(c)] if 0 <= int(c) < len(queries) else "unknown"
                for c in raw_class_ids
            ])

        batch["box_world"] = world_list
        batch["box_wkt"] = wkt_list
        batch["box_class"] = class_label_list
        return batch

# COMMAND ----------

# This class loads the SAM2 model and generates segmentation masks.
class SegmenterStep:

    SAM_MODEL = "sam2-hiera-small"
    WORKING_TYPE = np.uint32
    TARGET_TYPE = np.uint16
    MAX_VAL = np.iinfo(TARGET_TYPE).max
    SUB_BATCH_SIZE = 50
    KERNEL_SIZE = 5

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.sam = SamGeo2(model_id=self.SAM_MODEL, device=self.device, automatic=False)
        print("SegmenterStep initialized on device:", self.device)

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Applies morphological operations to improve mask quality.
        1. Closing: Fills small holes and bridges gaps (fixes fragmentation).
        2. Opening: Removes small speckles/noise (fixes false positives).
        """
        kernel = np.ones((self.KERNEL_SIZE, self.KERNEL_SIZE), np.uint8)
        
        cleaned = mask
        # cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        # cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Processes a batch to generate segmentation masks.
        - Uses the image and bounding boxes from the batch.
        - Generates a final mask overlay.
        - Adds the final mask to the batch.
        """
        masks_list = []
        for i in range(len(batch["image_path"])):
            rgb_array = batch["rgb_image_array"][i]
            original_np = batch["original_image_np"][i]
            raw_boxes = batch["boxes"][i]
            if raw_boxes is None or len(raw_boxes) == 0:
                boxes_np = np.empty((0, 4), dtype=np.float32)
            else:
                if isinstance(raw_boxes, (list, tuple, np.ndarray)):
                    processed_boxes = [
                        b.tolist() if hasattr(b, "tolist") else b 
                        for b in raw_boxes
                    ]
                    boxes_np = np.array(processed_boxes, dtype=np.float32)
                else:
                    boxes_np = np.atleast_2d(np.array(raw_boxes, dtype=np.float32))
            boxes_np = np.atleast_2d(boxes_np)
            num_boxes = boxes_np.shape[0]
            
            mask_overlay = np.zeros(original_np.shape[:2], dtype=self.WORKING_TYPE)

            if num_boxes > 0:
                image_pil = Image.fromarray(rgb_array)
                self.sam.set_image(image_pil)

                for k in range(0, num_boxes, self.SUB_BATCH_SIZE):
                    sub_boxes = boxes_np[k:k+self.SUB_BATCH_SIZE]
                    masks, _, _ = self.sam.predict(
                        boxes=sub_boxes,
                        multimask_output=False,
                        return_results=True
                        )
                
                    if masks.ndim == 4 and masks.shape[1] == 1:
                        masks = masks.squeeze(1)

                    for j in range(masks.shape[0]):
                        mask_id = k + j + 1
                        single_mask = (masks[j] > 0).astype(np.uint8) * 255
                        cleaned_single = self._clean_mask(single_mask)
                        mask_overlay[cleaned_single > 0] = mask_id

            if mask_overlay.max() > self.MAX_VAL:
                print(f"Too many objects for type: {self.TARGET_TYPE().dtype.name}")
                mask_overlay = np.clip(mask_overlay, 0, self.MAX_VAL)
            encoded_mask = mask_overlay.astype(self.TARGET_TYPE)

            success, encoded_image = cv2.imencode('.png', encoded_mask)
            masks_list.append(encoded_image.tobytes() if success else b"")

        batch["mask"] = masks_list
        return batch

# COMMAND ----------

# This class converts each image's PNG-encoded segmentation mask into a list
# of CRS-space WKT polygons -- one per distinct mask object, transformed from
# pixel space using the per-image affine. Mirrors the customer's pandas UDF.
class MaskPolygonizeStep:
    def __init__(self):
        pass

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        from rasterio.features import shapes as rio_shapes
        from rasterio.transform import Affine
        from shapely.geometry import shape as shapely_shape

        wkts_per_image = []
        for i in range(len(batch["image_path"])):
            mask_bytes = batch["mask"][i]
            polys = []

            if mask_bytes:
                nparr = np.frombuffer(mask_bytes, np.uint8)
                decoded = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                if decoded is not None:
                    object_ids = np.unique(decoded[decoded > 0])
                    if len(object_ids) > 0:
                        affine = Affine(*list(batch["transform"][i])[:6])
                        for obj_id in object_ids:
                            single_obj = (decoded == obj_id).astype(np.uint8)
                            for geom, val in rio_shapes(single_obj, transform=affine):
                                if val == 1:
                                    polys.append(shapely_shape(geom).wkt)

            wkts_per_image.append(polys)

        batch["mask_wkt"] = wkts_per_image
        return batch

# COMMAND ----------

# Merged segmentation + polygonisation step.
#
# Why merge: Ray Data fails to pass a `mask` (bytes) column between two
# successive `map_batches` actors -- pyarrow's ArrowPythonObjectScalar.as_py()
# does not accept the `maps_as_pydicts` kwarg Ray's newer code passes during
# the to_numpy conversion. By doing both SAM2 inference AND polygonisation in
# the same `__call__`, the encoded mask never leaves Python and Ray never
# tries to round-trip it through Arrow.
class SegmentAndPolygonizeStep:
    SAM_MODEL = "sam2-hiera-small"
    WORKING_TYPE = np.uint32
    TARGET_TYPE = np.uint16
    MAX_VAL = np.iinfo(TARGET_TYPE).max
    SUB_BATCH_SIZE = 50

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sam = SamGeo2(model_id=self.SAM_MODEL, device=self.device, automatic=False)
        print("SegmentAndPolygonizeStep initialized on device:", self.device)

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        from rasterio.features import shapes as rio_shapes
        from rasterio.transform import Affine
        from shapely.geometry import shape as shapely_shape

        masks_list = []
        wkts_list = []
        for i in range(len(batch["image_path"])):
            rgb_array = batch["rgb_image_array"][i]
            original_np = batch["original_image_np"][i]
            raw_boxes = batch["boxes"][i]
            transform = list(batch["transform"][i])[:6]

            # Coerce boxes -> (N, 4) float32
            if raw_boxes is None or len(raw_boxes) == 0:
                boxes_np = np.empty((0, 4), dtype=np.float32)
            else:
                processed_boxes = [
                    b.tolist() if hasattr(b, "tolist") else b for b in raw_boxes
                ]
                boxes_np = np.array(processed_boxes, dtype=np.float32)
            boxes_np = np.atleast_2d(boxes_np)
            num_boxes = boxes_np.shape[0]

            mask_overlay = np.zeros(original_np.shape[:2], dtype=self.WORKING_TYPE)
            if num_boxes > 0:
                image_pil = Image.fromarray(rgb_array)
                self.sam.set_image(image_pil)
                for k in range(0, num_boxes, self.SUB_BATCH_SIZE):
                    # samgeo2.predict only forwards `boxes` to SAM2 if it's a
                    # Python list -- numpy arrays are silently treated as None.
                    # Converting via .tolist() ensures the prompt actually lands.
                    sub_boxes = boxes_np[k:k + self.SUB_BATCH_SIZE].tolist()
                    masks, _, _ = self.sam.predict(
                        boxes=sub_boxes,
                        multimask_output=False,
                        return_results=True,
                    )
                    if masks.ndim == 4 and masks.shape[1] == 1:
                        masks = masks.squeeze(1)
                    for j in range(masks.shape[0]):
                        mask_id = k + j + 1
                        single_mask = (masks[j] > 0).astype(np.uint8)
                        mask_overlay[single_mask > 0] = mask_id

            if mask_overlay.max() > self.MAX_VAL:
                mask_overlay = np.clip(mask_overlay, 0, self.MAX_VAL)
            encoded_mask = mask_overlay.astype(self.TARGET_TYPE)

            # PNG-encode for storage / visualisation.
            success, encoded_image = cv2.imencode('.png', encoded_mask)
            masks_list.append(encoded_image.tobytes() if success else b"")

            # Polygonise directly from the uint16 mask -- no PNG round-trip.
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

        batch["mask"] = masks_list
        batch["mask_wkt"] = wkts_list
        return batch

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set `text_prompt`to whatever you want to detect

# COMMAND ----------

text_prompt = "pier"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pick the bounding-box detector
# MAGIC
# MAGIC Three open-vocabulary detectors are wired in:
# MAGIC
# MAGIC - `"owlv2"` -- `google/owlv2-large-patch14-ensemble`. Best stock results
# MAGIC   on this aerial AOI; tends to miss thin elongated piers.
# MAGIC - `"grounding_dino"` -- `IDEA-Research/grounding-dino-base`. Broader text
# MAGIC   matching; needs higher thresholds and is prone to false positives on
# MAGIC   industrial buildings in top-down views.
# MAGIC - `"omdet"` -- `omlab/omdet-turbo-swin-tiny-hf`. Fastest; weakest recall
# MAGIC   on this dataset.

# COMMAND ----------

BBOX_MODEL = "owlv2"  # one of: "owlv2", "grounding_dino", "omdet"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Disable segmentation when only bounding boxes are needed
# MAGIC
# MAGIC Set `RUN_SEGMENTATION = False` to skip the SAM2 stage during precision/recall
# MAGIC evaluation. The pipeline still emits pixel boxes, world-space boxes and WKT
# MAGIC polygons so the customer can compute IoU against ground truth.

# COMMAND ----------

RUN_SEGMENTATION = True

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set `source_dir` to the path of your Volume that stores your list of .tifs

# COMMAND ----------

CATALOG = "stuart"
SCHEMA = "tce"
VOLUME = "imagery"

# COMMAND ----------

image_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/ThamesAOI/Ortho_RGBN_P00084612_20090823_20090823_20cm_resTQ57nw.ecw"

data = spark.createDataFrame([(image_path,)], ["image_path"])\
  .withColumn("text_prompt", F.lit(text_prompt))

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
    concurrency=(1, 1 + max_worker_nodes),
    batch_size=4,
    num_gpus=0.75, # Request 50% of a GPU (~20GB on an A100/40GB)
)

# Step 3: BoxGeometryStep (CPU-only, lightweight)
# Reproject pixel boxes to source-CRS coordinates using the per-image affine
# and emit WKT polygons for downstream IoU evaluation.
ds = ds.map_batches(
    BoxGeometryStep,
    concurrency=(1, 3 * (1 + max_worker_nodes)),
    batch_size=8,
    num_cpus=1,
)

# Segmentation does NOT run inside this Ray pipeline -- it is split into a
# second Ray pipeline (further down) that consumes the dissolved cluster
# bboxes from Spark, so SAM2 sees clean inputs rather than the noisy raw
# detections.

# `boxes` and `box_world` are list-of-lists per row; Ray serialises them as
# opaque BINARY in Delta which breaks downstream `arrays_zip`. WKT is enough
# for IoU evaluation, so drop the numeric forms before write.
ds = ds.drop_columns(["rgb_image_array", "original_image_np", "boxes", "box_world", "class_ids"])

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

spark.sql(f"DROP TABLE IF EXISTS {tref}")

ds.write_databricks_table(tref, mode='overwrite')

# COMMAND ----------

spark.table(tref).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Per-box geometry table for IoU evaluation
# MAGIC
# MAGIC The per-image inference table holds detection arrays per row. Below we
# MAGIC explode it into one row per detected bounding box and promote the WKT
# MAGIC polygon to a Databricks `geometry` column in EPSG:27700 (British
# MAGIC National Grid). The customer joins this against their ground truth and
# MAGIC computes IoU per box (`st_intersection` / `st_area`) or at image level
# MAGIC (`st_union_agg`) -- exactly as in their reference notebook.

# COMMAND ----------

# Source imagery for this evaluation is supplied in EPSG:27700. Update if the
# `crs` column in the inference table reports a different CRS.
SRID = 27700
boxes_table = f"{tref}_boxes"

per_box = (
    spark.table(tref)
        # Ray serialises numeric arrays as STRUCT<data, shape> (its tensor
        # extension type). Pull `.data` out so it behaves like a plain array.
        .withColumn("box_confidence", F.col("box_confidence").getField("data"))
        .select(
            "image_path",
            "crs",
            "transform",
            F.posexplode(
                F.arrays_zip(
                    F.col("box_wkt").alias("box_wkt"),
                    F.col("box_class").alias("box_class"),
                    F.col("box_confidence").alias("box_confidence"),
                )
            ).alias("box_idx", "_b"),
        )
        .select("image_path", "crs", "transform", "box_idx", "_b.*")
        .withColumn("geometry", DBF.st_geomfromtext("box_wkt", SRID))
)

spark.sql(f"DROP TABLE IF EXISTS {boxes_table}")
per_box.write.mode("overwrite").saveAsTable(boxes_table)
display(spark.table(boxes_table))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dissolve overlapping detections into clusters
# MAGIC
# MAGIC Treats the detection set as a graph (node = box, edge = "intersects")
# MAGIC and replaces each connected component with the union polygon and its
# MAGIC envelope. The envelope is what `SegmenterStep` would consume as a SAM2
# MAGIC box prompt; the polygon is kept for visualisation and any future
# MAGIC polygon-vs-polygon comparison against ground truth.
# MAGIC
# MAGIC Each cluster also carries `n_boxes`, `mean_confidence`, and
# MAGIC `max_confidence` from the original detections that fell inside it.

# COMMAND ----------

clusters_table = f"{tref}_clusters"

CLUSTER_BUFFER_METRES = 10

clusters = spark.sql(f"""
    WITH dissolved AS (
        -- Buffer each detection by {CLUSTER_BUFFER_METRES}m before unioning so
        -- nearby-but-not-touching detections fall into the same cluster. Group
        -- by image_path so each cluster carries the image it came from --
        -- needed by the downstream segmentation pipeline.
        SELECT image_path,
               st_union_agg(st_buffer(geometry, {CLUSTER_BUFFER_METRES})) AS multi
        FROM {boxes_table}
        GROUP BY image_path
    ),
    components AS (
        SELECT d.image_path,
               t.cluster_id,
               st_geometryn(d.multi, t.cluster_id + 1) AS cluster_geom
        FROM dissolved d
        LATERAL VIEW explode(sequence(0, CAST(st_numgeometries(d.multi) AS INT) - 1)) t AS cluster_id
    )
    -- GEOMETRY columns are not orderable, so we GROUP BY (image_path,
    -- cluster_id) only and pull the (single) cluster_geom through with
    -- `any_value`.
    SELECT
        c.image_path,
        c.cluster_id,
        any_value(c.cluster_geom) AS cluster_geom,
        st_envelope(any_value(c.cluster_geom)) AS cluster_bbox,
        count(b.box_confidence) AS n_boxes,
        avg(b.box_confidence) AS mean_confidence,
        max(b.box_confidence) AS max_confidence
    FROM components c
    LEFT JOIN {boxes_table} b
        ON c.image_path = b.image_path
        AND st_intersects(c.cluster_geom, b.geometry)
    GROUP BY c.image_path, c.cluster_id
    ORDER BY c.image_path, c.cluster_id
""")

spark.sql(f"DROP TABLE IF EXISTS {clusters_table}")
clusters.write.mode("overwrite").saveAsTable(clusters_table)
display(spark.table(clusters_table))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline 2: Segmentation on dissolved cluster bboxes
# MAGIC
# MAGIC The dissolved `cluster_bbox` envelopes are clean inputs for SAM2 -- one
# MAGIC bbox per physical pier rather than the original swarm of overlapping
# MAGIC detections. This pipeline re-loads each image, converts the world-space
# MAGIC cluster envelopes to pixel coordinates via the per-image affine, runs
# MAGIC SAM2, and polygonises the resulting masks back to world-space WKT.
# MAGIC
# MAGIC Skipped entirely when `RUN_SEGMENTATION = False`.

# COMMAND ----------

if RUN_SEGMENTATION:
    # Build the per-image segmentation input: one row per image. We pass each
    # bbox coord as its own flat array<double> column rather than a nested
    # array<array<double>>, because Ray's from_spark trips over the latter
    # (pyarrow ArrowPythonObjectScalar incompatibility with `maps_as_pydicts`).
    seg_input_df = spark.sql(f"""
        SELECT
            image_path,
            collect_list(st_xmin(cluster_bbox)) AS xmins,
            collect_list(st_ymin(cluster_bbox)) AS ymins,
            collect_list(st_xmax(cluster_bbox)) AS xmaxs,
            collect_list(st_ymax(cluster_bbox)) AS ymaxs
        FROM {clusters_table}
        GROUP BY image_path
    """)
    display(seg_input_df)

# COMMAND ----------

# Converts world-space cluster envelopes to pixel-space [x1, y1, x2, y2]
# using each row's affine transform, and writes them to `batch["boxes"]` so
# SegmenterStep can consume them as SAM2 prompts.
class WorldToPixelBoxesStep:
    def __init__(self):
        pass

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        boxes_per_image = []
        for i in range(len(batch["image_path"])):
            xmins = list(batch["xmins"][i])
            ymins = list(batch["ymins"][i])
            xmaxs = list(batch["xmaxs"][i])
            ymaxs = list(batch["ymaxs"][i])
            a, _b, c, _d, e, f = list(batch["transform"][i])[:6]
            pixel_boxes = []
            for w_xmin, w_ymin, w_xmax, w_ymax in zip(xmins, ymins, xmaxs, ymaxs):
                # Inverse of an axis-aligned (north-up) affine. b == d == 0.
                p_xmin = (w_xmin - c) / a
                p_xmax = (w_xmax - c) / a
                # e is negative for north-up imagery, so the world's ymax (top)
                # maps to the smaller pixel y, and ymin (bottom) to the larger.
                p_y_top    = (w_ymax - f) / e
                p_y_bottom = (w_ymin - f) / e
                pixel_boxes.append([p_xmin, p_y_top, p_xmax, p_y_bottom])
            boxes_per_image.append(pixel_boxes)

            # One-shot diagnostic on the first row -- prints the affine, image
            # shape, and the world->pixel translation of the first three boxes,
            # so we can verify the bboxes truly land on the river piers in
            # pixel space.
            if i == 0:
                img_shape = batch["original_image_np"][i].shape
                print(f"[WorldToPixel] image_path = {batch['image_path'][i]}")
                print(f"[WorldToPixel] image shape (H, W, B) = {img_shape}")
                print(f"[WorldToPixel] affine (a, b, c, d, e, f) = "
                      f"({a}, {_b}, {c}, {_d}, {e}, {f})")
                print(f"[WorldToPixel] first {min(3, len(xmins))} world bboxes -> pixel:")
                for k in range(min(3, len(xmins))):
                    print(f"    world  [{xmins[k]:.1f}, {ymins[k]:.1f}, "
                          f"{xmaxs[k]:.1f}, {ymaxs[k]:.1f}]")
                    print(f"    pixel  [{pixel_boxes[k][0]:.1f}, {pixel_boxes[k][1]:.1f}, "
                          f"{pixel_boxes[k][2]:.1f}, {pixel_boxes[k][3]:.1f}]")

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

    # Convert cluster_bboxes_world -> pixel-space `boxes` ready for SAM2.
    seg_ds = seg_ds.map_batches(
        WorldToPixelBoxesStep,
        concurrency=(1, 3 * (1 + max_worker_nodes)),
        batch_size=8,
        num_cpus=1,
    )

    # SAM2 segmentation + in-step polygonisation. Merged into a single
    # map_batches actor so the bytes-typed `mask` column never has to be
    # passed between Ray actors (pyarrow ArrowPythonObjectScalar can't be
    # converted via the modern Ray->numpy code path).
    seg_ds = seg_ds.map_batches(
        SegmentAndPolygonizeStep,
        concurrency=(1, 1 + max_worker_nodes),
        batch_size=8,
        num_gpus=0.4,
    )

    seg_ds = seg_ds.drop_columns(
        ["rgb_image_array", "original_image_np", "boxes",
         "xmins", "ymins", "xmaxs", "ymaxs"]
    )

# COMMAND ----------

if RUN_SEGMENTATION:
    segmentation_table = f"{tref}_segmentation"
    spark.sql(f"DROP TABLE IF EXISTS {segmentation_table}")
    seg_ds.write_databricks_table(segmentation_table, mode="overwrite")
    display(spark.table(segmentation_table))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Per-segment geometry table
# MAGIC
# MAGIC Explodes the per-image `mask_wkt` array into one row per polygon, with
# MAGIC an EPSG:27700 `geometry` column ready for IoU evaluation.

# COMMAND ----------

if RUN_SEGMENTATION:
    segments_table = f"{tref}_segments"

    per_segment = (
        spark.table(segmentation_table)
            .where("size(mask_wkt) > 0")
            .select(
                "image_path",
                "crs",
                "transform",
                F.posexplode("mask_wkt").alias("seg_idx", "seg_wkt"),
            )
            .withColumn("geometry", DBF.st_geomfromtext("seg_wkt", SRID))
    )

    spark.sql(f"DROP TABLE IF EXISTS {segments_table}")
    per_segment.write.mode("overwrite").saveAsTable(segments_table)
    display(spark.table(segments_table))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation against ground truth
# MAGIC
# MAGIC Compares detections in `inference_results_boxes` to the ground-truth
# MAGIC polygons in `${CATALOG}.${SCHEMA}.ground_truth` (populated by the
# MAGIC `load_ground_truth` notebook).
# MAGIC
# MAGIC The detector emits axis-aligned bounding boxes, while ground-truth
# MAGIC polygons trace each pier's actual shape, so we take the **envelope**
# MAGIC (minimum bounding rectangle) of each ground-truth polygon before
# MAGIC computing IoU. This is the apples-to-apples comparison.
# MAGIC
# MAGIC Reports:
# MAGIC
# MAGIC - **Image-level IoU** -- the dissolve-and-intersect approach from the
# MAGIC   customer's reference notebook (single number).
# MAGIC - **Per-box matching** -- for each detection, the IoU of its best
# MAGIC   ground-truth match; same for each ground-truth bbox. From those,
# MAGIC   precision, recall and F1 are computed at `IOU_MATCH_THRESHOLD`.

# COMMAND ----------

GROUND_TRUTH_TABLE = f"{CATALOG}.{SCHEMA}.ground_truth"
IOU_MATCH_THRESHOLD = 0.5  # min IoU to count a detection as a true positive

# COMMAND ----------

if not spark.catalog.tableExists(GROUND_TRUTH_TABLE):
    print(f"Ground truth table {GROUND_TRUTH_TABLE} not found.")
    print("Run the `load_ground_truth` notebook (in src/) before evaluating.")
else:
    # Ground-truth polygons trace the precise pier shape, but the dissolved
    # cluster bboxes are axis-aligned. Take the envelope of each GT polygon
    # so the comparison is bbox-to-bbox.
    spark.sql(f"""
        CREATE OR REPLACE TEMP VIEW gt_bbox AS
        SELECT gt_id, st_envelope(geometry) AS geometry
        FROM {GROUND_TRUTH_TABLE}
    """)

    # Image-level IoU -- dissolve both sets, then compute IoU of the unions.
    # Detections side uses the dissolved `cluster_bbox` so duplicates from the
    # raw per-box table do not double-count.
    image_iou = spark.sql(f"""
        WITH gt AS (SELECT st_union_agg(geometry) AS geom FROM gt_bbox),
             pr AS (SELECT st_union_agg(cluster_bbox) AS geom FROM {clusters_table}),
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

    # Pairwise IoU across every (cluster, ground-truth-bbox) pair that intersect.
    spark.sql(f"""
        CREATE OR REPLACE TEMP VIEW pair_iou AS
        SELECT
            gt.gt_id,
            det.image_path,
            det.cluster_id,
            st_area(st_intersection(gt.geometry, det.cluster_bbox)) /
                NULLIF(
                    st_area(gt.geometry) + st_area(det.cluster_bbox)
                        - st_area(st_intersection(gt.geometry, det.cluster_bbox)),
                    0
                ) AS iou
        FROM gt_bbox gt
        CROSS JOIN {clusters_table} det
        WHERE st_intersects(gt.geometry, det.cluster_bbox)
    """)

    # Best matching IoU per cluster (precision side) and per ground truth (recall side).
    spark.sql(f"""
        CREATE OR REPLACE TEMP VIEW det_match AS
        SELECT det.image_path,
               det.cluster_id,
               COALESCE(MAX(p.iou), 0.0) AS best_iou
        FROM {clusters_table} det
        LEFT JOIN pair_iou p USING (image_path, cluster_id)
        GROUP BY det.image_path, det.cluster_id
    """)
    spark.sql(f"""
        CREATE OR REPLACE TEMP VIEW gt_match AS
        SELECT gt.gt_id,
               COALESCE(MAX(p.iou), 0.0) AS best_iou
        FROM gt_bbox gt
        LEFT JOIN pair_iou p USING (gt_id)
        GROUP BY gt.gt_id
    """)

    det_count, det_tp = spark.sql(
        f"SELECT COUNT(*), SUM(CASE WHEN best_iou >= {IOU_MATCH_THRESHOLD} THEN 1 ELSE 0 END) FROM det_match"
    ).first()
    gt_count, gt_tp = spark.sql(
        f"SELECT COUNT(*), SUM(CASE WHEN best_iou >= {IOU_MATCH_THRESHOLD} THEN 1 ELSE 0 END) FROM gt_match"
    ).first()

    det_count, gt_count = int(det_count or 0), int(gt_count or 0)
    det_tp, gt_tp = int(det_tp or 0), int(gt_tp or 0)

    precision = det_tp / det_count if det_count else 0.0
    recall    = gt_tp  / gt_count  if gt_count  else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"IoU match threshold  : {IOU_MATCH_THRESHOLD}")
    print(f"Ground-truth count   : {gt_count}")
    print(f"Cluster count        : {det_count}")
    print(f"Clusters matched     : {det_tp}    -> precision = {precision:.3f}")
    print(f"Ground-truth matched : {gt_tp}    -> recall    = {recall:.3f}")
    print(f"F1                   : {f1:.3f}")

# COMMAND ----------

if RUN_SEGMENTATION:
    from base64 import b64encode

    png_bytes = spark.table(segmentation_table).select("mask").sort("image_path", ascending=True).first().mask

    displayHTML(f'<img src="data:image/png;base64,{b64encode(png_bytes).decode("ascii")}" height=300/>')
else:
    print("RUN_SEGMENTATION=False -- no mask column to preview.")