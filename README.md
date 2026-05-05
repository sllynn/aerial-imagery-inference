# Ray Inference — Databricks Asset Bundle

Wraps `src/ray_inference.py` — a Databricks notebook that runs OWLv2
open-vocabulary bounding-box detection (with optional SAM2 segmentation)
over GeoTIFF / ECW imagery on a Ray-on-Spark cluster — as a deployable
bundle for the Azure workspace
`https://adb-984752964297111.11.azuredatabricks.net`.

The pipeline emits two Unity Catalog tables:

- `stuart.tce.inference_results` — one row per source image, with the
  per-image affine transform, CRS, and the list of detected
  `box_wkt` polygons.
- `stuart.tce.inference_results_boxes` — one row per detected box, with
  an EPSG:27700 `geometry` column ready for IoU evaluation against the
  customer's BNG ground truth.

For evaluation, the bundle deploys `data/piers.csv` (one quoted WKT
POLYGON per row, no header) alongside the notebooks. A separate notebook
`src/load_ground_truth.py` reads it and writes the polygons to
`stuart.tce.ground_truth`. The inference notebook then joins the two
tables and reports image-level IoU plus per-box precision/recall/F1 at
a configurable IoU threshold.

## Layout

```
.
├── databricks.yml                  # bundle root + dev target
├── requirements.txt                # runtime deps (installed via uv)
├── excludes.txt                    # packages NOT to install (DBR provides)
├── resources/
│   ├── ray_inference.job.yml       # inference job + cluster binding
│   └── load_ground_truth.job.yml   # ground-truth loader job
├── data/
│   └── piers.csv                   # ground-truth polygons (quoted WKT per row)
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

# Refresh auth for the target workspace (profile already exists as `arm`):
databricks auth login --profile arm
```

## Validate, deploy, run

```bash
databricks bundle validate -t dev -p arm
databricks bundle deploy   -t dev -p arm

# One-off: load the ground-truth polygons before the first evaluation.
databricks bundle run load_ground_truth -t dev -p arm

# Each inference run also produces the IoU / precision / recall / F1 cells.
databricks bundle run ray_inference -t dev -p arm
```

## Cluster

Bound to the existing all-purpose cluster `0501-151131-jp5lcmfz`. To
switch to an ephemeral job cluster, replace `existing_cluster_id` in
`resources/ray_inference.job.yml` with a `job_cluster_key` and a
`job_clusters:` block.

The cluster must be configured with:

- **Spark configs** (Compute → Edit → Advanced → Spark):
  - `spark.databricks.pyspark.dataFrameChunk.enabled true` — required
    by `ray.data.from_spark`.
  - `spark.task.resource.gpu.amount 0` — leaves the GPU to Ray.
- **Init script** that installs the GDAL ECW driver plus
  `gdal-config` / GDAL dev headers (rasterio is built from source
  against this system GDAL — see `requirements.txt`).

## Dependency strategy

`requirements.txt` is installed at notebook startup with `uv`:

```
%pip install uv
%sh uv pip install \
  --no-binary rasterio \
  --excludes ../excludes.txt \
  -r ../requirements.txt
%restart_python
```

- `--no-binary rasterio` forces a source build so rasterio links against
  the cluster's GDAL (which carries the ECW plugin).
- `--excludes ../excludes.txt` skips re-installing `torch`,
  `torchvision`, `transformers`, and `huggingface-hub` — DBR ML's
  pre-installed versions are matched to the cluster's CUDA driver and
  shadowing them with PyPI wheels causes GPU-mode failure.

## Pipeline

The Ray-on-Spark pipeline runs four steps on the dataset of imagery
paths:

1. **`PreProcessorStep`** — opens each GeoTIFF/ECW with rasterio,
   percentile-scales each band to uint8, and attaches the per-image
   `crs` and `transform` to the batch.
2. **`BBoxPredictorStep`** — slices each image into 640×640 tiles
   (50 % overlap) via `supervision.InferenceSlicer`, runs OWLv2
   (`google/owlv2-large-patch14-ensemble`) at the configured
   `text_prompt`, and returns pixel-space bounding boxes after
   per-slice area filtering and cross-slice NMS.
3. **`BoxGeometryStep`** — reprojects pixel boxes to source-CRS
   coordinates using the stored affine and emits matching WKT POLYGON
   strings (the customer's reference logic).
4. **`SegmenterStep`** *(skipped when `RUN_SEGMENTATION=False`, the
   default)* — runs SAM2 on each box to produce a per-image segmentation
   mask, encoded as PNG bytes.

After the Ray write, a Spark cell explodes per-image rows into one
row per box and lifts `box_wkt` to a Databricks `geometry` column
(EPSG:27700) for the customer's IoU evaluation.

## Tunable knobs

In `src/ray_inference.py`:

- `text_prompt` — comma-separated open-vocabulary queries
  (default `"pier,jetty"`).
- `BBOX_MODEL` — selects the bounding-box detector. One of:
  - `"owlv2"` — `google/owlv2-large-patch14-ensemble` (default; best
    stock results on this AOI).
  - `"grounding_dino"` — `IDEA-Research/grounding-dino-base` (broader
    text matching; needs higher thresholds).
  - `"omdet"` — `omlab/omdet-turbo-swin-tiny-hf` (fastest; weakest
    recall on this dataset).
  Per-model thresholds live in `BBoxPredictorStep._MODELS`.
- `RUN_SEGMENTATION` — toggle the SAM2 stage.
- `CATALOG` / `SCHEMA` / `VOLUME` — change if the workspace differs.
- `image_path` (or revert to a `dbutils.fs.ls(source_dir)` listing) —
  one-shot single-file input, currently pointing at
  `Ortho_RGBN_P00084612_..._20cm_resTQ57nw.ecw`.
- Slicer settings inside `BBoxPredictorStep.__call__` — `slice_wh`,
  `overlap_wh`, `iou_threshold` for cross-slice NMS, and the
  post-slicer pixel-dimension cap (default `960 × 0.8`).
- The post-write `SRID` (default `27700`) — update if source imagery
  is supplied in a different CRS.
- `GROUND_TRUTH_TABLE` and `IOU_MATCH_THRESHOLD` (in the evaluation
  cells) — the table holding the customer's reference polygons and the
  IoU above which a detection counts as a true positive (default `0.5`).

In `src/load_ground_truth.py`:

- `GROUND_TRUTH_TABLE` — Delta table the loader populates (must match
  the value in the inference notebook).
- `SRID` — CRS of the supplied WKT (default `27700`).
- The CSV path is resolved automatically from the notebook's own
  location to `data/piers.csv` in the deployed bundle.
