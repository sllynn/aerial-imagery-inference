# Scalable Object Detection and Segmentation on Aerial Imagery

Wraps `src/ray_inference.py` — a Databricks notebook that runs
open-vocabulary bounding-box detection (OWLv2 / Grounding DINO / OmDet)
and SAM2 / SAM3 segmentation over GeoTIFF / ECW orthorectified, georeferenced 
imagery on a Ray-on-Spark cluster — as a deployable bundle for a Databricks
workspace (set the workspace host in `databricks.yml`).

The pipeline emits up to three Unity Catalog tables, all keyed by
`image_path`:

- `${CATALOG}.${SCHEMA}.inference_results_boxes` — one row per detected
  bounding box, with `box_pixel` (pixel-space `[x1, y1, x2, y2]`),
  `box_wkt` (world-space WKT POLYGON), and a Databricks `geometry`
  column in EPSG:27700 (BNG by default).
- `${CATALOG}.${SCHEMA}.inference_results_clusters` — only when
  `RUN_DISSOLVE=true`. Per cluster envelope produced by the
  buffer-and-union dissolve, with `cluster_pixel`, `cluster_bbox_wkt`,
  and a `cluster_bbox` `geometry` column.
- `${CATALOG}.${SCHEMA}.inference_results_segments_<segmenter>` — only
  when `RUN_SEGMENTATION=true`. One row per polygon traced from the
  segmentation mask, with `seg_wkt` and a `geometry` column. The
  `<segmenter>` suffix is `samgeo2` or `samgeo3`, so both backends'
  outputs persist side-by-side.

For evaluation, the bundle deploys `data/piers.csv` (a QGIS-style
attribute-table export with a `WKT` column plus optional
`_predicate` / `SHAPE_Leng` / `SHAPE_Area` columns) alongside the
notebooks. A separate notebook `src/load_ground_truth.py` reads only
the `WKT` column and writes the polygons to
`${CATALOG}.${SCHEMA}.ground_truth`. The inference notebook then
evaluates both bbox detections (with `st_envelope` of the ground-truth
polygons for an apples-to-apples bbox comparison) and segmentation
polygons (against the raw ground truth) at the configured
`IOU_MATCH_THRESHOLD`.

## Source imagery

The orthographic imagery used by this notebook is downloaded from the
UK government's open Environment Survey portal:

> https://environment.data.gov.uk/survey

Pick a region of interest on the map, then download the four-band
**Orthographic Imagery** layer — the files are delivered as
`.ecw` (Enhanced Compression Wavelet, ERDAS's proprietary format).
Drop the `.ecw` files into the Unity Catalog volume pointed at by the
`CATALOG` / `SCHEMA` / `VOLUME` widgets
(`/Volumes/<catalog>/<schema>/<volume>/...`) and set the `image_path`
widget to the file you want to process.

Reading ECW from rasterio on Databricks requires the GDAL ECW plugin,
which is not bundled with rasterio's binary wheels. The cluster init
script must compile and install it; the
[`rasterio-ecw-databricks`](https://github.com/sllynn/rasterio-ecw-databricks)
repository documents the exact build steps (download the ERDAS ECW
SDK, compile the GDAL plugin against the cluster's GDAL version, and
drop the resulting `.so` into `GDAL_DRIVER_PATH`). Use that recipe to
produce the init script referenced under "Cluster" below.

## Layout

```
.
├── databricks.yml                  # bundle root + dev target
├── requirements.txt                # top-level runtime deps
├── requirements.lock               # full transitive pins (uv pip compile)
├── excludes.txt                    # packages NOT to install (DBR provides)
├── resources/
│   ├── ray_inference.job.yml       # inference job + parameters + cluster binding
│   └── load_ground_truth.job.yml   # ground-truth loader job
├── data/
│   └── piers.csv                   # ground-truth polygons (WKT + QGIS attrs)
├── src/
│   ├── ray_inference.py            # the inference + evaluation notebook
│   └── load_ground_truth.py        # loads data/piers.csv into a Delta table
├── requirements-dev.txt            # local CLI helpers
└── .venv/                          # local Python virtualenv
```

## One-time setup

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt

# Refresh auth for the target workspace. Substitute your own profile name.
databricks auth login --profile <profile>
```

## Validate, deploy, run

Substitute `<profile>` with the Databricks CLI profile pointing at
your workspace.

```bash
databricks bundle validate -t dev -p <profile>
databricks bundle deploy   -t dev -p <profile>

# One-off: load the ground-truth polygons before the first evaluation.
databricks bundle run load_ground_truth -t dev -p <profile>

# Each inference run produces both bbox and segmentation IoU summaries.
databricks bundle run ray_inference -t dev -p <profile>
```

Override any widget at run time via `--params`:

```bash
# Skip the dissolve and use SAM3 for segmentation:
databricks bundle run ray_inference -t dev -p <profile> \
  --params RUN_DISSOLVE=false,SEGMENTER_VERSION=samgeo3

# Different image with grounding_dino, no segmentation:
databricks bundle run ray_inference -t dev -p <profile> \
  --params image_path=/Volumes/.../some.ecw,BBOX_MODEL=grounding_dino,RUN_SEGMENTATION=false
```

## Cluster

Bound to an existing all-purpose cluster via the `existing_cluster_id`
field in `resources/ray_inference.job.yml`. Replace it with your own
cluster id, or swap to an ephemeral job cluster by substituting a
`job_cluster_key` plus a `job_clusters:` block.

The cluster must be configured with:

- **Spark configs** (Compute → Edit → Advanced → Spark):
  - `spark.databricks.pyspark.dataFrameChunk.enabled true` — required
    by `ray.data.from_spark`.
  - `spark.task.resource.gpu.amount 0` — leaves the GPU to Ray.
- **Init script** that installs the GDAL ECW driver plus
  `gdal-config` / GDAL dev headers (rasterio is built from source
  against this system GDAL). Build steps are documented at
  [`sllynn/rasterio-ecw-databricks`](https://github.com/sllynn/rasterio-ecw-databricks).
- **Databricks secret** `<secret-scope>/hf_token` — required for
  `samgeo3`, whose SAM3 weights are gated on Hugging Face. The
  notebook reads it via
  `dbutils.secrets.get(scope="<secret-scope>", key="hf_token")` and
  forwards it to every Ray actor through `runtime_env`. Update the
  scope in `src/ray_inference.py` to match your workspace.

## Dependency strategy

`requirements.lock` is installed at notebook startup with `uv`:

```
%pip install uv
%sh uv pip install \
  --no-binary rasterio \
  --excludes ../excludes.txt \
  -r ../requirements.lock
%sh uv pip install \
  --reinstall --no-deps \
  "sam3 @ git+https://github.com/facebookresearch/sam3.git"
%restart_python
```

- `requirements.lock` is generated from `requirements.txt` with
  ```
  uv pip compile \
    --python-version 3.12 \
    --python-platform x86_64-unknown-linux-gnu \
    requirements.txt -o requirements.lock
  ```
  Re-run this whenever `requirements.txt` changes.
- `--no-binary rasterio` forces a source build so rasterio links
  against the cluster's GDAL (which carries the ECW plugin).
- `--excludes ../excludes.txt` skips re-installing `torch`,
  `torchvision`, `triton`, and `flash-attn` — DBR ML's pre-installed
  versions are matched to the cluster's CUDA driver and shadowing them
  with PyPI wheels causes GPU-mode failure.
- `sam3` is installed separately from Meta's GitHub source with
  `--no-deps`, because the PyPI `sam3==0.0.1` is an unrelated stub and
  Meta's transitive deps would re-pull a CUDA-bound torch.
  Pure-Python prerequisites (`timm`, `ftfy`, `iopath`, `pycocotools`)
  are listed in `requirements.txt` so the lock includes them.

## Pipeline

The notebook runs **two Ray-on-Spark pipelines** off the same input
image, both bracketed by `setup_ray_cluster(...)`:

**Pipeline 1 — bounding boxes (always runs):**

1. **`PreProcessorStep`** — opens each GeoTIFF/ECW with rasterio,
   percentile-scales each band to uint8, and attaches the per-image
   `crs` and `transform` to the batch.
2. **`BBoxPredictorStep`** — slices each image into 960×960 tiles
   (480 px overlap) via `supervision.InferenceSlicer`, runs the
   selected detector at the configured `text_prompt`, applies
   per-slice area filtering and cross-slice NMS
   (`iou_threshold=0.30`), and emits pixel-space boxes plus class
   indices and confidences.
3. **`BoxGeometryStep`** — reprojects pixel boxes to source-CRS
   coordinates using the stored affine, emits matching WKT POLYGON
   strings, and (when `RUN_DISSOLVE=true`) buffer-and-unions
   overlapping AABBs via Python union-find to produce one cluster
   envelope per physical structure. The cluster step also emits each
   envelope's pixel coordinates so Pipeline 2 can prompt SAM directly
   without redoing the world↔pixel inversion.
4. **`ExplodeBoxesStep`** / **`ExplodeClustersStep`** — flatten
   per-image arrays into per-detection rows in Ray (no Spark
   `posexplode`). Each branch writes to a staging Delta table; a tiny
   Spark `CREATE TABLE … AS SELECT *, st_geomfromtext(...) AS geometry`
   step lifts the WKT to a queryable `geometry` column.

**Pipeline 2 — segmentation (skipped when `RUN_SEGMENTATION=false`):**

Sourced from `inference_results_clusters` when `RUN_DISSOLVE=true`,
otherwise from `inference_results_boxes`. Per-image input rows carry
the pixel-space xmins/ymins/xmaxs/ymaxs as four flat `array<double>`
columns (avoids Ray's pyarrow trip on `array<array<double>>`).

1. **`PreProcessorStep`** (re-used) — re-loads the image to recover
   `rgb_image_array` and `original_image_np` for SAM input and mask
   shape.
2. **`PixelBoxAssemblerStep`** — zips the four flat arrays back into
   a per-image list of 4-tuples for the segmenter.
3. **`SegmentAndPolygonizeStep`** (SAM2) or
   **`SegmentAndPolygonizeStepSAM3`** (SAM3, `meta` backend in
   interactive mode). Both inherit from `_BaseSegmentAndPolygonizeStep`,
   which owns the box coercion, the per-image label-overlay loop, and
   the `rasterio.features.shapes` polygonisation. Subclasses just load
   the model and implement `_predict_masks`. Inference and
   polygonisation run in the same actor — encoded masks never cross a
   Ray Arrow boundary.
4. **`ExplodeSegmentsStep`** — same per-image-to-per-row flatten as
   the bbox path. Spark adds the `geometry` column.

## Evaluation

`evaluate_against_ground_truth(...)` runs once per pipeline:

- **Bbox eval** uses `st_envelope` of each ground-truth polygon so the
  comparison is bbox-to-bbox.
- **Segment eval** uses raw ground-truth polygons (segmentation
  outputs already trace the shape).

Both report:

- **Image-level IoU** — `st_intersection / st_area` on the dissolved
  unions of detections and ground truth.
- **Per-detection precision / recall / F1** — for each detection the
  best matching ground-truth IoU; same per ground-truth row;
  detections (or GT polygons) with `best_iou ≥ IOU_MATCH_THRESHOLD`
  are true positives.

Each call returns a metrics dict (`bbox_metrics`, `seg_metrics`)
ready for any further comparison cells.

## Tunable knobs (notebook widgets / job parameters)

All run-time knobs are exposed as Databricks notebook widgets and
mirrored as job parameters in
`resources/ray_inference.job.yml`. They are all overridable via
`databricks bundle run ray_inference --params key=val,...`.

| Widget                | Default                                                          | Notes |
|-----------------------|------------------------------------------------------------------|-------|
| `text_prompt`         | `pier`                                                           | Comma-separated open-vocabulary queries. |
| `BBOX_MODEL`          | `owlv2`                                                          | One of `owlv2`, `grounding_dino`, `omdet`. Per-model thresholds live in `BBoxPredictorStep._MODELS`. |
| `RUN_SEGMENTATION`    | `true`                                                           | Toggles Pipeline 2. |
| `SEGMENTER_VERSION`   | `samgeo2`                                                        | `samgeo2` (SAM2 / `sam2-hiera-small`) or `samgeo3` (SAM3 / `facebook/sam3`, transformers backend). |
| `RUN_DISSOLVE`        | `true`                                                           | Buffer-and-union overlapping bbox detections (10 m buffer in `BoxGeometryStep.BUFFER_METRES`). Helpful for AOIs like Thames where OWLv2 produces swarms of overlapping detections; can muddy small/close structures (e.g. Norfolk). |
| `CATALOG`/`SCHEMA`/`VOLUME` | (set per workspace)                                        | Unity Catalog locations for input imagery and output tables. Defaults live in `resources/ray_inference.job.yml` and the bottom widget cell of the notebook — change both to point at your own catalog/schema/volume. |
| `image_path`          | (set per run)                                                    | Full Volumes path to the source image. |
| `SRID`                | `27700`                                                          | EPSG code of the source imagery. |
| `IOU_MATCH_THRESHOLD` | `0.5`                                                            | Min IoU above which a detection counts as a true positive. |

The widgets themselves are declared in the very last cell of the
notebook, after a `dbutils.notebook.exit()` — so a "Run All" never
re-declares them. On jobs the widget bar is populated via
`base_parameters` in the YAML; for interactive use, run that bottom
cell once to refresh the bar.

Source-only knobs (still constants in the notebook, not widgets):

- Slicer settings inside `BBoxPredictorStep.__call__` — `slice_wh`,
  `overlap_wh`, `iou_threshold`, and the post-slicer pixel-dimension
  cap (default `960 × 0.8`).
- `BoxGeometryStep.BUFFER_METRES` — buffer applied to each AABB
  before union-find dissolve.
- Per-model detection thresholds in `BBoxPredictorStep._MODELS`.

In `src/load_ground_truth.py`:

- `CATALOG`, `SCHEMA` — Unity Catalog location of the
  `ground_truth` table (must match the inference notebook).
- `SRID` — CRS of the supplied WKT (default `27700`).
- The CSV path is resolved automatically from the notebook's own
  location to `data/piers.csv` in the deployed bundle.
