# Databricks notebook source
# MAGIC %md
# MAGIC # Load Ground Truth WKT into a Delta Table
# MAGIC
# MAGIC Reads a CSV of WKT polygons (one quoted POLYGON per line, no header)
# MAGIC deployed alongside the bundle at `data/piers.csv` and writes them to a
# MAGIC Delta table with an EPSG:27700 `geometry` column. The inference notebook
# MAGIC joins this table against `inference_results_boxes` to compute per-image
# MAGIC IoU, precision, recall, and F1.

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

from pyspark.databricks.sql import functions as DBF

# Workspace files are visible to plain Python on the driver but not to Spark's
# distributed reader, which would prefix the path with `dbfs:` and fail. So we
# read the CSV in pure Python here, then hand the parsed rows to Spark.
wkts = []
with open(GROUND_TRUTH_FILE, "r") as fh:
    for raw in fh:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        # Strip the wrapping double quotes the CSV format adds around each WKT.
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1]
        wkts.append(s)

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
