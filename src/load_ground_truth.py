# Databricks notebook source
# MAGIC %md
# MAGIC # Load Ground Truth WKT into a Delta Table
# MAGIC
# MAGIC Reads a CSV of WKT polygons deployed alongside the bundle at
# MAGIC `data/piers.csv` and writes them to a Delta table with an
# MAGIC EPSG:27700 `geometry` column. The inference notebook joins this
# MAGIC table against `inference_results_boxes` to compute per-image IoU,
# MAGIC precision, recall, and F1.
# MAGIC
# MAGIC Expected CSV shape: a header row containing a `WKT` column followed
# MAGIC by any number of attribute columns (e.g. QGIS's standard
# MAGIC `WKT,_predicate,SHAPE_Leng,SHAPE_Area`). Only the `WKT` column is
# MAGIC used; everything else is ignored.

# COMMAND ----------

CATALOG = "stuart"
SCHEMA = "tce"

GROUND_TRUTH_TABLE = f"{CATALOG}.{SCHEMA}.ground_truth"

# Source CRS for the supplied WKT. Pier orthos for this engagement are in BNG.
SRID = 27700

# COMMAND ----------

# Resolve the absolute Workspace path to data/piers.csv from the notebook's own
# location. The bundle uploads `src/load_ground_truth.py` to .../files/src/ and
# `data/piers.csv` to .../files/data/, so the file is one directory above the
# notebook's parent.
import os
_nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
GROUND_TRUTH_FILE = "/Workspace" + os.path.dirname(os.path.dirname(_nb)) + "/data/piers.csv"
print(f"Reading ground truth from: {GROUND_TRUTH_FILE}")

# COMMAND ----------

import csv

from pyspark.databricks.sql import functions as DBF

# Workspace files are visible to plain Python on the driver but not to Spark's
# distributed reader, which would prefix the path with `dbfs:` and fail. So we
# read the CSV in pure Python here, then hand the parsed rows to Spark.
# `csv.DictReader` handles the quoted WKT cell (which contains commas) and
# the auxiliary attribute columns from QGIS exports without us having to
# strip quotes by hand.
wkts = []
with open(GROUND_TRUTH_FILE, "r", newline="") as fh:
    reader = csv.DictReader(fh)
    if reader.fieldnames is None or "WKT" not in reader.fieldnames:
        raise ValueError(
            f"{GROUND_TRUTH_FILE}: expected a header row containing a "
            f"`WKT` column; got {reader.fieldnames!r}."
        )
    for row in reader:
        wkt = (row.get("WKT") or "").strip()
        if wkt:
            wkts.append(wkt)

print(f"Read {len(wkts)} ground-truth WKT polygons from file.")

gt = (
    spark.createDataFrame(
        [(i, w) for i, w in enumerate(wkts)],
        schema="gt_id long, wkt string",
    )
    .withColumn("geometry", DBF.st_geomfromtext("wkt", SRID))
    .select("gt_id", "wkt", "geometry")
)

spark.sql(f"DROP TABLE IF EXISTS {GROUND_TRUTH_TABLE}")
gt.write.mode("overwrite").saveAsTable(GROUND_TRUTH_TABLE)
print(f"Wrote {spark.table(GROUND_TRUTH_TABLE).count()} ground-truth polygons to {GROUND_TRUTH_TABLE}")
display(spark.table(GROUND_TRUTH_TABLE))
