# Databricks notebook source
# MAGIC %md
# MAGIC # Postprocessing
# MAGIC
# MAGIC Notebook to vectorise the raster output

# COMMAND ----------

# MAGIC %pip install rasterio shapely affine leafmap==0.20.0 geopandas==1.1.1 folium==0.13.0 mapclassify==2.10.0 opencv-python==4.12.0.88
# MAGIC %restart_python

# COMMAND ----------

import pandas as pd
import numpy as np
import pickle
import rasterio.features
import cv2
from shapely.geometry import shape
from affine import Affine

import pyspark.sql.functions as F
import pyspark.databricks.sql.functions as DBF
from pyspark.sql.types import ArrayType, StringType

# COMMAND ----------

CATALOG = "stuart"
SCHEMA = "tce"

# COMMAND ----------

@F.pandas_udf(ArrayType(StringType()))
def polygonize_mask(masks: pd.Series, transforms: pd.Series) -> pd.Series:
  """
  A Pandas UDF that converts segmentation masks into an array of WKT polygons.

  Args:
      masks (pd.Series): A Series of 2D NumPy arrays representing the segmentation masks.
      transforms (pd.Series): A Series of lists representing the Affine transform parameters.

  Returns:
      pd.Series: A Series where each element is a list of WKT strings, with each string representing one polygon found in the corresponding mask.
  """
  results = []
  # Iterate through each row of the batch (mask and its corresponding transform)
  for mask, transform in zip(masks, transforms):
    row_polygons = []

    nparr = np.frombuffer(mask, np.uint8)
    decoded_mask = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if decoded_mask is None:
        results.append([])
        continue

    mask = decoded_mask

    # mask = mask.reshape((5000, 5000))

    # Find the unique object IDs in the mask, excluding 0 (background)
    object_ids = np.unique(mask[mask > 0])

    transform = Affine(*transform[:6])
    
    if len(object_ids) == 0:
      results.append([])
      continue

    for obj_id in object_ids:
      # Create a boolean mask for the current object ID
      single_object_mask = (mask == obj_id).astype(np.uint8)

      # Extract shapes from the boolean mask. This returns a generator.
      # The transform correctly maps pixel coordinates to geo coordinates.
      shapes = rasterio.features.shapes(
          single_object_mask, 
          transform=transform
      )

        # Convert each shape into a WKT polygon string
      for geom, val in shapes:
        if val == 1: # Ensure we are processing the foreground shape
          # Convert the GeoJSON-like dictionary to a Shapely object
          polygon = shape(geom)
          # Append the WKT representation to our list for this row
          row_polygons.append(polygon.wkt)

    results.append(row_polygons)
      
  return pd.Series(results)

# COMMAND ----------

# MAGIC %md
# MAGIC Change to your table from the Ray output

# COMMAND ----------

table_name = "inference_results"
output_table = f"{CATALOG}.{SCHEMA}.{table_name}"

data = spark.table(output_table)

# COMMAND ----------

polygons_df = data.select("mask", "transform", "crs", "text_prompt").withColumn(
    "polygons_wkt", 
    polygonize_mask(F.col("mask"), F.col("transform.data"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC These steps assume the Spatial SQL preview is enabled

# COMMAND ----------

exploded_polygons_df = polygons_df\
  .withColumn("polygon_wkt", F.explode(F.col("polygons_wkt")))\
  .select("text_prompt", "crs", "polygon_wkt").distinct()\
  .withColumn("polygon_id", F.monotonically_increasing_id())\
  .withColumn("geom_27700", F.expr("st_geomfromwkt(polygon_wkt, 27700)"))\
  .withColumn("geom", F.expr("st_transform(geom_27700, 4326)"))\
  .withColumn("polygon_wkt", F.expr("st_astext(geom)"))\
  .withColumn("area", F.expr("st_area(geom)"))\
  .withColumn("geojson", F.expr("st_asgeojson(geom)"))\
  .drop("geom", "geom_27700")

# COMMAND ----------

# MAGIC %md
# MAGIC Change to your table name

# COMMAND ----------

polygon_table_name = "polygons"
polygon_tref = f"{CATALOG}.{SCHEMA}.{polygon_table_name}"

exploded_polygons_df.withColumn("id", F.monotonically_increasing_id()).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(polygon_tref)

# COMMAND ----------

# MAGIC %md
# MAGIC (optionally) visualise some of them in leafmap

# COMMAND ----------

indexed_polys_df = spark.table(polygon_tref).select("polygon_id", "polygon_wkt").withColumn("h3_index", F.explode(DBF.h3_coverash3(F.col("polygon_wkt"), 8))).withColumn("geom", DBF.st_geomfromwkt(F.col("polygon_wkt"), 4326))

join_expr = F.expr("(i1.h3_index = i2.h3_index) AND (st_intersects(i1.geom, i2.geom)) AND (i1.polygon_id <= i2.polygon_id)")

joined_df = indexed_polys_df.alias("i1").join(indexed_polys_df.alias("i2"), join_expr, "inner")

unioned_polys = (
  joined_df
  .withColumn("union_geom", DBF.st_union(F.col("i1.geom"), F.col("i2.geom")))
  .groupBy("i1.polygon_id")
  .agg(DBF.st_union_agg("union_geom").alias("union_geom"))
  .withColumn("union_geom", DBF.st_simplify("union_geom", F.lit(1e-6)))
  .where(~DBF.st_isempty("union_geom"))
  .where(DBF.st_area(F.col("union_geom").cast("GEOGRAPHY(4326)")) > 10)
  )

display(unioned_polys)

# COMMAND ----------

df = unioned_polys.withColumn("union_geom", DBF.st_astext("union_geom")).toPandas()
display(df)

# COMMAND ----------

df.count()

# COMMAND ----------

import geopandas as gpd
from shapely import wkt

df['geom'] = df['union_geom'].apply(wkt.loads)
gpd.GeoDataFrame(df[:100], geometry="geom", crs="EPSG:4326").explore(tiles="Esri.WorldImagery")
