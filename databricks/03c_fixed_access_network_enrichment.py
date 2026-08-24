# Databricks notebook source
from pyspark.sql import functions as F

CUSTOMER_TABLE = "workspace.telecom_gold.dim_customer"
NETWORK_TABLE = "workspace.telecom_gold.fact_network_events"

CUSTOMER_BACKUP = (
    "workspace.telecom_gold."
    "dim_customer_backup_before_service_fix"
)

NETWORK_BACKUP = (
    "workspace.telecom_gold."
    "fact_network_events_backup_before_cabinet_fix"
)

CUSTOMER_TEMP = (
    "workspace.telecom_gold."
    "dim_customer_service_fixed_temp"
)

NETWORK_TEMP = (
    "workspace.telecom_gold."
    "fact_network_events_cabinet_temp"
)

CABINET_PERFORMANCE_TABLE = (
    "workspace.telecom_gold."
    "network_performance_by_cabinet"
)

customers = spark.table(CUSTOMER_TABLE)

display(
    customers
    .groupBy("internet_service")
    .count()
    .orderBy(F.col("count").desc())
)

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CUSTOMER_BACKUP}
USING DELTA
AS
SELECT *
FROM {CUSTOMER_TABLE}
""")

customer_fixed = customers.withColumn(
    "internet_service",
    F.when(
        F.lower(
            F.trim(F.col("internet_service"))
        ).isin(
            "no",
            "none",
            "no internet service"
        ),
        "No Internet Service"
    )
    .when(
        F.col("internet_service").isNull(),
        "Unknown"
    )
    .otherwise(F.col("internet_service"))
)

(
    customer_fixed.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(CUSTOMER_TEMP)
)

spark.sql(f"""
CREATE OR REPLACE TABLE {CUSTOMER_TABLE}
USING DELTA
AS
SELECT *
FROM {CUSTOMER_TEMP}
""")

spark.sql(f"DROP TABLE IF EXISTS {CUSTOMER_TEMP}")

display(
    spark.table(CUSTOMER_TABLE)
    .groupBy("internet_service")
    .count()
    .orderBy(F.col("count").desc())
)

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {NETWORK_BACKUP}
USING DELTA
AS
SELECT *
FROM {NETWORK_TABLE}
""")

network = spark.table(NETWORK_TABLE)

customer_service = (
    spark.table(CUSTOMER_TABLE)
    .select(
        "customer_key",
        "internet_service"
    )
    .dropDuplicates(["customer_key"])
)

network_with_service = (
    network
    .join(
        customer_service,
        on="customer_key",
        how="left"
    )
    .withColumn(
        "access_technology",
        F.when(
            F.lower(
                F.trim(F.col("internet_service"))
            ) == "dsl",
            "DSL"
        )
        .when(
            F.lower(
                F.trim(F.col("internet_service"))
            ).isin(
                "fiber optic",
                "fiber",
                "fiber-optic"
            ),
            "Fiber Optic"
        )
    )

    # Remove customers without fixed internet
    .filter(
        F.col("access_technology").isNotNull()
    )

    # Short region code for Cabinet ID
    .withColumn(
        "_region_code",
        F.when(F.col("region") == "Cairo", "CAI")
         .when(F.col("region") == "Giza", "GIZ")
         .when(F.col("region") == "Alexandria", "ALX")
         .when(F.col("region") == "Delta", "DLT")
         .when(F.col("region") == "Upper Egypt", "UPR")
         .otherwise("UNK")
    )

    # Stable cabinet assignment
    .withColumn(
        "_cabinet_number",
        (
            F.pmod(
                F.xxhash64(
                    F.col("customer_id").cast("string"),
                    F.col("region"),
                    F.col("access_technology")
                ),
                F.lit(100)
            ) + 1
        ).cast("int")
    )

    .withColumn(
        "cabinet_id",
        F.concat(
            F.lit("CAB-"),
            F.when(
                F.col("access_technology") == "DSL",
                F.lit("DSL")
            ).otherwise(F.lit("FBR")),
            F.lit("-"),
            F.col("_region_code"),
            F.lit("-"),
            F.lpad(
                F.col("_cabinet_number").cast("string"),
                3,
                "0"
            )
        )
    )

    .withColumn(
        "cabinet_type",
        F.when(
            F.col("access_technology") == "DSL",
            "DSL Cabinet"
        ).otherwise("Fiber Cabinet")
    )
)

if "_gold_created_at" in network.columns:
    network_with_service = network_with_service.withColumn(
        "_gold_created_at",
        F.current_timestamp()
    )

network_enriched = network_with_service.select(
    *network.columns,
    "access_technology",
    "cabinet_id",
    "cabinet_type"
)

source_count = network.count()
enriched_count = network_enriched.count()

print(f"Original network events: {source_count:,}")
print(f"Eligible DSL/Fiber events: {enriched_count:,}")
print(
    "Excluded No Internet Service events: "
    f"{source_count - enriched_count:,}"
)

(
    network_enriched.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(NETWORK_TEMP)
)

spark.sql(f"""
CREATE OR REPLACE TABLE {NETWORK_TABLE}
USING DELTA
AS
SELECT *
FROM {NETWORK_TEMP}
""")

spark.sql(f"DROP TABLE IF EXISTS {NETWORK_TEMP}")

print("Fixed-access network fact updated successfully.")

# COMMAND ----------

fixed_network = spark.table(NETWORK_TABLE)

cabinet_performance = (
    fixed_network
    .groupBy(
        "cabinet_id",
        "cabinet_type",
        "access_technology",
        "region"
    )
    .agg(
        F.countDistinct("event_id")
         .alias("network_events"),

        F.sum(
            F.when(
                F.col("event_type") != "Normal",
                1
            ).otherwise(0)
        ).alias("network_incidents"),

        F.countDistinct(
            F.when(
                F.col("event_type") != "Normal",
                F.col("customer_id")
            )
        ).alias("affected_customers"),

        F.round(
            F.avg("latency_ms"), 2
        ).alias("average_latency_ms"),

        F.round(
            F.avg("packet_loss_pct"), 2
        ).alias("average_packet_loss_pct"),

        F.round(
            F.avg("download_speed_mbps"), 2
        ).alias("average_download_speed_mbps"),

        F.sum("outage_minutes")
         .alias("total_outage_minutes")
    )
    .withColumn(
        "incident_rate_pct",
        F.round(
            F.col("network_incidents")
            / F.col("network_events")
            * 100,
            2
        )
    )
    .withColumn(
        "_gold_created_at",
        F.current_timestamp()
    )
)

(
    cabinet_performance.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(CABINET_PERFORMANCE_TABLE)
)

display(
    cabinet_performance
    .filter(F.col("network_events") >= 20)
    .orderBy(
        F.col("incident_rate_pct").desc(),
        F.col("network_incidents").desc()
    )
    .limit(20)
)

print("Cabinet performance table created successfully.")