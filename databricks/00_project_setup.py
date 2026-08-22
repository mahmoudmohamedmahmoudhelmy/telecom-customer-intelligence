# Databricks notebook source
print("Telecom Customer Intelligence Platform")
print("Databricks environment is ready")

spark.sql("SHOW CATALOGS").show(truncate=False)


# COMMAND ----------

#  Schemas of Medallion Architecture 

spark.sql("""
CREATE SCHEMA IF NOT EXISTS workspace.telecom_bronze
COMMENT 'Raw ingested telecom data'
""")

spark.sql("""
CREATE SCHEMA IF NOT EXISTS workspace.telecom_silver
COMMENT 'Cleaned, validated and integrated telecom data'
""")

spark.sql("""
CREATE SCHEMA IF NOT EXISTS workspace.telecom_gold
COMMENT 'Business-ready analytics and ML tables'
""")

# managed volume creation fot the file sources
spark.sql("""
CREATE VOLUME IF NOT EXISTS workspace.telecom_bronze.raw_files
COMMENT 'Landing zone for raw telecom source files'
""")

print("Bronze, Silver and Gold schemas created successfully.")

display(spark.sql("SHOW SCHEMAS IN workspace"))

# COMMAND ----------

raw_volume_path = "/Volumes/workspace/telecom_bronze/raw_files"

uploaded_files = dbutils.fs.ls(raw_volume_path)

display(uploaded_files)

# COMMAND ----------

