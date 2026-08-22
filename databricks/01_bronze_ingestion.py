# Databricks notebook source
from datetime import datetime
from uuid import uuid4

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
    TimestampType
)

# Unity Catalog configuration
CATALOG = "workspace"
BRONZE_SCHEMA = "telecom_bronze"

# Raw landing-zone volume
RAW_VOLUME = f"/Volumes/{CATALOG}/{BRONZE_SCHEMA}/raw_files"

# Source paths
CUSTOMERS_PATH = f"{RAW_VOLUME}/telco_customers.csv"
BILLING_PATH = f"{RAW_VOLUME}/billing_transactions.json"
COMPLAINTS_PATH = f"{RAW_VOLUME}/customer_complaints.csv"
NETWORK_PATH = f"{RAW_VOLUME}/network_events.json"

# One identifier for this ingestion run
BATCH_ID = str(uuid4())

print(f"Batch ID: {BATCH_ID}")
print(f"Raw volume: {RAW_VOLUME}")

# COMMAND ----------

def add_ingestion_metadata(df, source_system):
    business_columns = df.columns

    return (
        df
        .withColumn("_source_system", F.lit(source_system))
        .withColumn("_source_file", F.col("_metadata.file_path"))
        .withColumn("_ingestion_timestamp", F.current_timestamp())
        .withColumn("_batch_id", F.lit(BATCH_ID))
        .withColumn(
            "_record_hash",
            F.sha2(
                F.concat_ws(
                    "||",
                    *[
                        F.coalesce(
                            F.col(column).cast("string"),
                            F.lit("")
                        )
                        for column in business_columns
                    ]
                ),
                256
            )
        )
    )

# COMMAND ----------

customer_schema = StructType([
    StructField("customerID", StringType(), False),
    StructField("gender", StringType(), True),
    StructField("SeniorCitizen", IntegerType(), True),
    StructField("Partner", StringType(), True),
    StructField("Dependents", StringType(), True),
    StructField("tenure", IntegerType(), True),
    StructField("PhoneService", StringType(), True),
    StructField("MultipleLines", StringType(), True),
    StructField("InternetService", StringType(), True),
    StructField("OnlineSecurity", StringType(), True),
    StructField("OnlineBackup", StringType(), True),
    StructField("DeviceProtection", StringType(), True),
    StructField("TechSupport", StringType(), True),
    StructField("StreamingTV", StringType(), True),
    StructField("StreamingMovies", StringType(), True),
    StructField("Contract", StringType(), True),
    StructField("PaperlessBilling", StringType(), True),
    StructField("PaymentMethod", StringType(), True),
    StructField("MonthlyCharges", DoubleType(), True),
    StructField("TotalCharges", DoubleType(), True),
    StructField("Churn", StringType(), True)
])

# COMMAND ----------

customers_raw = (
    spark.read
    .format("csv")
    .option("header", True)
    .option("mode", "PERMISSIVE")
    .schema(customer_schema)
    .load(CUSTOMERS_PATH)
)

customers_bronze = add_ingestion_metadata(
    customers_raw,
    "CRM"
)

display(customers_bronze.limit(10))

# COMMAND ----------

CUSTOMERS_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.customers_raw"

(
    customers_bronze.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(CUSTOMERS_TABLE)
)

print(f"Created Delta table: {CUSTOMERS_TABLE}")
print(f"Record count: {spark.table(CUSTOMERS_TABLE).count():,}")

# COMMAND ----------

customer_quality = spark.sql("""
SELECT
    COUNT(*) AS total_records,
    COUNT(DISTINCT customerID) AS unique_customers,
    SUM(CASE WHEN customerID IS NULL THEN 1 ELSE 0 END) AS null_customer_ids,
    SUM(CASE WHEN MonthlyCharges IS NULL THEN 1 ELSE 0 END) AS null_monthly_charges,
    SUM(CASE WHEN Churn NOT IN ('Yes', 'No') OR Churn IS NULL THEN 1 ELSE 0 END)
        AS invalid_churn_values
FROM workspace.telecom_bronze.customers_raw
""")

display(customer_quality)

# COMMAND ----------

from pyspark.sql import functions as F

from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
    TimestampType
)

# COMMAND ----------

billing_schema = StructType([
    StructField("invoice_id", StringType(), False),
    StructField("customerID", StringType(), False),
    StructField("billing_month", TimestampType(), True),
    StructField("MonthlyCharges", DoubleType(), True),
    StructField("usage_charges", DoubleType(), True),
    StructField("discount_amount", DoubleType(), True),
    StructField("tax_amount", DoubleType(), True),
    StructField("total_amount", DoubleType(), True),
    StructField("PaymentMethod", StringType(), True),
    StructField("payment_status", StringType(), True),
    StructField("payment_date", TimestampType(), True)
])

# COMMAND ----------

billing_raw = (
    spark.read
    .format("json")
    .option("mode", "PERMISSIVE")
    .schema(billing_schema)
    .load(BILLING_PATH)
)

billing_bronze = add_ingestion_metadata(
    billing_raw,
    "Billing System"
)

display(billing_bronze.limit(10))

# COMMAND ----------

BILLING_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.billing_transactions_raw"

(
    billing_bronze.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(BILLING_TABLE)
)

print(f"Created Delta table: {BILLING_TABLE}")

billing_count = spark.table(BILLING_TABLE).count()
print(f"Record count: {billing_count:,}")

# COMMAND ----------

billing_quality = spark.sql("""
SELECT
    COUNT(*) AS total_transactions,
    COUNT(DISTINCT invoice_id) AS unique_invoices,
    SUM(CASE WHEN invoice_id IS NULL THEN 1 ELSE 0 END)
        AS null_invoice_ids,
    SUM(CASE WHEN customerID IS NULL THEN 1 ELSE 0 END)
        AS null_customer_ids,
    SUM(CASE WHEN total_amount < 0 THEN 1 ELSE 0 END)
        AS negative_amounts,
    SUM(
        CASE
            WHEN payment_status NOT IN ('Paid', 'Late', 'Unpaid')
            THEN 1 ELSE 0
        END
    ) AS invalid_payment_status,
    ROUND(SUM(total_amount), 2) AS total_billed_revenue,
    ROUND(AVG(total_amount), 2) AS average_invoice_value
FROM workspace.telecom_bronze.billing_transactions_raw
""")

display(billing_quality)

# COMMAND ----------

complaints_schema = StructType([
    StructField("complaint_id", StringType(), False),
    StructField("customerID", StringType(), False),
    StructField("opened_at", TimestampType(), True),
    StructField("category", StringType(), True),
    StructField("priority", StringType(), True),
    StructField("status", StringType(), True),
    StructField("channel", StringType(), True),
    StructField("resolution_hours", DoubleType(), True),
    StructField("satisfaction_score", IntegerType(), True)
])

# COMMAND ----------

complaints_raw = (
    spark.read
    .format("csv")
    .option("header", True)
    .option("mode", "PERMISSIVE")
    .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
    .schema(complaints_schema)
    .load(COMPLAINTS_PATH)
)

complaints_bronze = add_ingestion_metadata(
    complaints_raw,
    "Complaints System"
)

display(complaints_bronze.limit(10))

# COMMAND ----------

COMPLAINTS_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.customer_complaints_raw"

(
    complaints_bronze.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(COMPLAINTS_TABLE)
)

print(f"Created Delta table: {COMPLAINTS_TABLE}")

complaints_count = spark.table(COMPLAINTS_TABLE).count()
print(f"Record count: {complaints_count:,}")

# COMMAND ----------

complaints_quality = spark.sql("""
SELECT
    COUNT(*) AS total_complaints,
    COUNT(DISTINCT complaint_id) AS unique_complaints,

    SUM(
        CASE WHEN complaint_id IS NULL
        THEN 1 ELSE 0 END
    ) AS null_complaint_ids,

    SUM(
        CASE WHEN customerID IS NULL
        THEN 1 ELSE 0 END
    ) AS null_customer_ids,

    SUM(
        CASE WHEN resolution_hours < 0
        THEN 1 ELSE 0 END
    ) AS invalid_resolution_hours,

    SUM(
        CASE
            WHEN satisfaction_score NOT BETWEEN 1 AND 5
            THEN 1 ELSE 0
        END
    ) AS invalid_satisfaction_scores,

    SUM(
        CASE
            WHEN priority NOT IN ('Low', 'Medium', 'High', 'Critical')
            THEN 1 ELSE 0
        END
    ) AS invalid_priorities,

    ROUND(AVG(resolution_hours), 2) AS average_resolution_hours,
    ROUND(AVG(satisfaction_score), 2) AS average_satisfaction_score

FROM workspace.telecom_bronze.customer_complaints_raw
""")

display(complaints_quality)

# COMMAND ----------

NETWORK_STREAM_DIRECTORY = f"{RAW_VOLUME}/network_stream"

NETWORK_SOURCE_FILE = f"{RAW_VOLUME}/network_events.json"
NETWORK_STREAM_FILE = (
    f"{NETWORK_STREAM_DIRECTORY}/network_events_batch_001.json"
)

# Create a dedicated source directory for network-event files
dbutils.fs.mkdirs(NETWORK_STREAM_DIRECTORY)

# Copy the first network batch into the streaming directory
if not any(
    file.name == "network_events_batch_001.json"
    for file in dbutils.fs.ls(NETWORK_STREAM_DIRECTORY)
):
    dbutils.fs.cp(
        NETWORK_SOURCE_FILE,
        NETWORK_STREAM_FILE
    )

print(f"Streaming source: {NETWORK_STREAM_DIRECTORY}")

display(dbutils.fs.ls(NETWORK_STREAM_DIRECTORY))

# COMMAND ----------

spark.sql("""
CREATE VOLUME IF NOT EXISTS
workspace.telecom_bronze.streaming_checkpoints
COMMENT 'Structured Streaming checkpoints and Auto Loader schemas'
""")

CHECKPOINT_VOLUME = (
    "/Volumes/workspace/telecom_bronze/streaming_checkpoints"
)

NETWORK_CHECKPOINT_PATH = (
    f"{CHECKPOINT_VOLUME}/network_events_checkpoint"
)

NETWORK_SCHEMA_PATH = (
    f"{CHECKPOINT_VOLUME}/network_events_schema"
)

print(f"Checkpoint path: {NETWORK_CHECKPOINT_PATH}")
print(f"Schema path: {NETWORK_SCHEMA_PATH}")

# COMMAND ----------

network_schema = StructType([
    StructField("event_id", StringType(), False),
    StructField("customerID", StringType(), False),
    StructField("event_timestamp", TimestampType(), True),
    StructField("event_type", StringType(), True),
    StructField("region", StringType(), True),
    StructField("tower_id", StringType(), True),
    StructField("latency_ms", DoubleType(), True),
    StructField("packet_loss_pct", DoubleType(), True),
    StructField("download_speed_mbps", DoubleType(), True),
    StructField("outage_minutes", IntegerType(), True)
])

# COMMAND ----------

network_stream_raw = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", NETWORK_SCHEMA_PATH)
    .option("cloudFiles.schemaEvolutionMode", "rescue")
    .option("rescuedDataColumn", "_rescued_data")
    .schema(network_schema)
    .load(NETWORK_STREAM_DIRECTORY)
)

network_stream_bronze = add_ingestion_metadata(
    network_stream_raw,
    "Network Monitoring System"
)

print("Network streaming DataFrame created successfully.")
print(f"Is streaming: {network_stream_bronze.isStreaming}")

# COMMAND ----------

NETWORK_TABLE = (
    f"{CATALOG}.{BRONZE_SCHEMA}.network_events_raw"
)

network_query = (
    network_stream_bronze.writeStream
    .format("delta")
    .outputMode("append")
    .option(
        "checkpointLocation",
        NETWORK_CHECKPOINT_PATH
    )
    .trigger(availableNow=True)
    .toTable(NETWORK_TABLE)
)

network_query.awaitTermination()

print(f"Streaming ingestion completed: {NETWORK_TABLE}")

# COMMAND ----------

network_count = spark.table(NETWORK_TABLE).count()

print(f"Network event count: {network_count:,}")

display(
    spark.table(NETWORK_TABLE)
    .orderBy(F.col("event_timestamp").desc())
    .limit(10)
)

# COMMAND ----------

network_quality = spark.sql("""
SELECT
    COUNT(*) AS total_events,
    COUNT(DISTINCT event_id) AS unique_events,

    SUM(
        CASE WHEN event_id IS NULL
        THEN 1 ELSE 0 END
    ) AS null_event_ids,

    SUM(
        CASE WHEN customerID IS NULL
        THEN 1 ELSE 0 END
    ) AS null_customer_ids,

    SUM(
        CASE WHEN latency_ms < 0
        THEN 1 ELSE 0 END
    ) AS invalid_latency,

    SUM(
        CASE
            WHEN packet_loss_pct < 0
              OR packet_loss_pct > 100
            THEN 1 ELSE 0
        END
    ) AS invalid_packet_loss,

    SUM(
        CASE WHEN download_speed_mbps < 0
        THEN 1 ELSE 0 END
    ) AS invalid_download_speed,

    SUM(
        CASE WHEN outage_minutes < 0
        THEN 1 ELSE 0 END
    ) AS invalid_outage_duration,

    ROUND(AVG(latency_ms), 2) AS average_latency_ms,
    ROUND(AVG(packet_loss_pct), 2) AS average_packet_loss,
    ROUND(AVG(download_speed_mbps), 2) AS average_download_speed,

    SUM(
        CASE WHEN event_type = 'Outage'
        THEN 1 ELSE 0 END
    ) AS total_outage_events

FROM workspace.telecom_bronze.network_events_raw
""")

display(network_quality)

# COMMAND ----------

bronze_summary = spark.sql("""
SELECT 'customers_raw' AS table_name, COUNT(*) AS record_count
FROM workspace.telecom_bronze.customers_raw

UNION ALL

SELECT 'billing_transactions_raw', COUNT(*)
FROM workspace.telecom_bronze.billing_transactions_raw

UNION ALL

SELECT 'customer_complaints_raw', COUNT(*)
FROM workspace.telecom_bronze.customer_complaints_raw

UNION ALL

SELECT 'network_events_raw', COUNT(*)
FROM workspace.telecom_bronze.network_events_raw
""")

display(bronze_summary)