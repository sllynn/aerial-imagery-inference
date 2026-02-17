# Databricks notebook source
# MAGIC %pip install "segment-geospatial[samgeo2]==0.13.0" rasterio==1.4.3 supervision==0.27.0 "ray[data]==2.41.0" opencv-python==4.12.0.88 opencv-python-headless==4.12.0.88
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
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import rasterio
from PIL import Image
from typing import Dict

import pyspark.sql.functions as F

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

# This class loads the Owlv2 model and finds bounding boxes.
class BBoxPredictorStep:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bbox_model = "google/owlv2-large-patch14-ensemble"
        self.processor = Owlv2Processor.from_pretrained(bbox_model)
        self.model = Owlv2ForObjectDetection.from_pretrained(bbox_model).to(self.device)
        print("BBoxPredictorStep initialized on device:", self.device)

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Processes a batch of images and text prompts to find bounding boxes.
        - Converts NumPy image arrays to PIL Images.
        - Uses the Owlv2 model to predict bounding boxes.
        - Adds the predicted boxes to the batch.
        """
        boxes_list = []
        for i in range(len(batch["image_path"])):
            rgb_array = batch["rgb_image_array"][i]
            text_prompt = batch["text_prompt"][i]
            def callback(image_slice: np.ndarray) -> sv.Detections:
                # The slicer passes a numpy array. Convert to PIL for the transformer processor.
                # Note: PreProcessorStep already converted to RGB, so we don't need cvtColor here.
                image_pil = Image.fromarray(image_slice)
                
                inputs = self.processor(text=text_prompt, images=image_pil, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Post-process detection relative to the slice size
                target_sizes = torch.Tensor([image_pil.size[::-1]]).to(self.device)
                results = self.processor.post_process_grounded_object_detection(
                    outputs=outputs, 
                    target_sizes=target_sizes, 
                    threshold=0.3 # Low threshold before NMS
                )
                
                # Convert transformer results to supervision Detections
                return sv.Detections.from_transformers(results[0])

            # Initialize the Slicer
            slicer = sv.InferenceSlicer(
                callback=callback,
                slice_wh=(960, 960),
                overlap_wh=(240, 240),
                overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
                iou_threshold=0.5,
            )
            
            # Run inference on the full image (slicer handles the tiling loop)
            detections = slicer(rgb_array)

            # Add boxes (as a list of lists) to our list for the next step
            # sv.Detections.xyxy returns a numpy array of bounding boxes
            # boxes_list.append(detections.xyxy.tolist())
            boxes = detections.xyxy.astype(np.float32)
            boxes_list.append(boxes.tolist())

        batch["boxes"] = boxes_list
        return batch

# COMMAND ----------

# This class loads the SAM2 model and generates segmentation masks.
class SegmenterStep:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sam_model = "sam2-hiera-small"
        self.sam = SamGeo2(model_id=sam_model, device=self.device, automatic=False)
        print("SegmenterStep initialized on device:", self.device)

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Applies morphological operations to improve mask quality.
        1. Closing: Fills small holes and bridges gaps (fixes fragmentation).
        2. Opening: Removes small speckles/noise (fixes false positives).
        """
        kernel = np.ones((5, 5), np.uint8)
        
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
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
            boxes_input = batch["boxes"][i]
            if isinstance(boxes_input, np.ndarray):
                boxes_input = boxes_input.tolist()

            boxes = torch.tensor(boxes_input) # Convert list back to tensor

            h, w = rgb_array.shape[:2]
            mask_overlay = np.zeros((h, w), dtype=np.uint8)

            if boxes.nelement() > 0:
                image_pil = Image.fromarray(rgb_array)
                self.sam.set_image(image_pil)
                masks, _, _ = self.sam.predict(
                    boxes=boxes.numpy().tolist(),
                    multimask_output=False,
                    return_results=True
                    )
                
                if masks.ndim == 4 and masks.shape[1] == 1:
                    masks = masks.squeeze(1)

                mask_overlay = np.zeros_like(original_np[..., 0], dtype=np.uint8)
                for j, mask in enumerate(masks):
                    mask_overlay += ((mask > 0) * (j + 1)).astype(np.uint8)

            cleaned_mask = self._clean_mask(mask_overlay)

            success, encoded_image = cv2.imencode('.png', cleaned_mask.astype(np.uint8))
            if success:
                masks_list.append(encoded_image.tobytes())
            else:
                # Fallback: empty bytes if encoding fails (should never happen)
                masks_list.append(b"")

        batch["mask"] = masks_list
        return batch

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set `text_prompt`to whatever you want to detect

# COMMAND ----------

text_prompt = "pier"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set `source_dir` to the path of your Volume that stores your list of .tifs

# COMMAND ----------

CATALOG = "stuart"
SCHEMA = "tce"
VOLUME = "imagery"

# COMMAND ----------

source_dir = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/ThamesAOI/"


data = spark.createDataFrame(dbutils.fs.ls(source_dir))\
  .withColumn("image_path", F.expr("substring(path, 6, length(path))"))\
  .withColumn("text_prompt", F.lit(text_prompt))

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
    batch_size=4 * 5,
    num_gpus=0.75, # Request 50% of a GPU (~20GB on an A100/40GB)
)

# Step 3: SegmenterStep (Medium VRAM GPU task)
# We request the remaining portion of the GPU. Requesting slightly less than
# the remainder (e.g., 0.4 instead of 0.5) provides a safety margin for
# CUDA contexts and framework overhead.
ds = ds.map_batches(
    SegmenterStep,
    concurrency=(1, 1 + max_worker_nodes),
    batch_size=8 * 5,
    num_gpus=0.2, # Request 40% of a GPU (~16GB on an A100/40GB)
)

ds = ds.drop_columns(["rgb_image_array", "original_image_np"])

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

from base64 import b64encode

png_bytes = spark.table(tref).select("mask").sort("image_path", ascending=True).first().mask

displayHTML(f'<img src="data:image/png;base64,{b64encode(png_bytes).decode("ascii")}" height=300/>')
