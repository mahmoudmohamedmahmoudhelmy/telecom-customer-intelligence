# Databricks notebook source
from pyspark.sql import functions as F

LOCATION_TABLE = (
    "workspace.telecom_gold.dim_network_location"
)

NETWORK_TABLE = (
    "workspace.telecom_gold.fact_network_events"
)

CUSTOMER_TABLE = (
    "workspace.telecom_gold.dim_customer"
)

REGIONAL_TABLE = (
    "workspace.telecom_gold.network_performance_by_region"
)

EGYPT_BACKUP = (
    "workspace.telecom_gold."
    "fact_network_events_backup_before_city_enrichment"
)

CALIFORNIA_BACKUP = (
    "workspace.telecom_gold."
    "fact_network_events_backup_before_california_geo"
)

NETWORK_TEMP = (
    "workspace.telecom_gold."
    "fact_network_events_california_temp"
)

locations = [
    # Los Angeles Metro
    (101, "Los Angeles", "Los Angeles Metro",
     34.0522, -118.2437, 1.00),

    (102, "Long Beach", "Los Angeles Metro",
     33.7701, -118.1937, 1.08),

    (103, "Anaheim", "Los Angeles Metro",
     33.8366, -117.9143, 0.92),

    (104, "Riverside", "Los Angeles Metro",
     33.9806, -117.3755, 1.15),

    # Bay Area
    (201, "San Francisco", "Bay Area",
     37.7749, -122.4194, 0.88),

    (202, "San Jose", "Bay Area",
     37.3382, -121.8863, 0.82),

    (203, "Oakland", "Bay Area",
     37.8044, -122.2712, 1.08),

    # San Diego County
    (301, "San Diego", "San Diego County",
     32.7157, -117.1611, 0.90),

    (302, "Chula Vista", "San Diego County",
     32.6401, -117.0842, 1.05),

    # Central Valley
    (401, "Fresno", "Central Valley",
     36.7378, -119.7871, 1.05),

    (402, "Bakersfield", "Central Valley",
     35.3733, -119.0187, 1.18),

    (403, "Stockton", "Central Valley",
     37.9577, -121.2908, 1.12),

    # Sacramento and Northern California
    (501, "Sacramento", "Sacramento / North",
     38.5816, -121.4944, 1.00),

    (502, "Redding", "Sacramento / North",
     40.5865, -122.3917, 1.22)
]

location_columns = [
    "location_key",
    "city",
    "region",
    "latitude",
    "longitude",
    "quality_factor"
]

location_df = (
    spark.createDataFrame(
        locations,
        location_columns
    )
    .withColumn(
        "location_key",
        F.col("location_key").cast("long")
    )
    .withColumn("state", F.lit("California"))
    .withColumn("country", F.lit("United States"))
    .withColumn(
        "location_label",
        F.concat_ws(
            ", ",
            F.col("city"),
            F.col("state"),
            F.col("country")
        )
    )
)

(
    location_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(LOCATION_TABLE)
)

display(location_df)

# COMMAND ----------

# If the Egypt enrichment was executed, restore its clean backup.
# Otherwise, create a new recoverable backup.

if spark.catalog.tableExists(EGYPT_BACKUP):
    BASE_NETWORK_TABLE = EGYPT_BACKUP
    print("Using pre-Egypt-enrichment backup.")
else:
    spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CALIFORNIA_BACKUP}
    USING DELTA
    AS
    SELECT *
    FROM {NETWORK_TABLE}
    """)

    BASE_NETWORK_TABLE = CALIFORNIA_BACKUP
    print("California migration backup created.")

network_base = spark.table(BASE_NETWORK_TABLE)

customer_service = (
    spark.table(CUSTOMER_TABLE)
    .select(
        "customer_key",
        "internet_service"
    )
    .dropDuplicates(["customer_key"])
)

# Replace the existing regional labels
network_remapped = (
    network_base

    .withColumn(
        "region",
        F.when(
            F.col("region") == "Cairo",
            "Los Angeles Metro"
        )
        .when(
            F.col("region") == "Giza",
            "Bay Area"
        )
        .when(
            F.col("region") == "Alexandria",
            "San Diego County"
        )
        .when(
            F.col("region") == "Delta",
            "Central Valley"
        )
        .when(
            F.col("region") == "Upper Egypt",
            "Sacramento / North"
        )
        .otherwise(F.col("region"))
    )

    .join(
        customer_service,
        on="customer_key",
        how="left"
    )

    .withColumn(
        "_internet_service",
        F.lower(
            F.trim(F.col("internet_service"))
        )
    )

    # Fixed-broadband events only
    .filter(
        F.col("_internet_service").isin(
            "dsl",
            "fiber optic",
            "fiber",
            "fiber-optic"
        )
    )

    # Consistent city assignment for each customer
    .withColumn(
        "_u_city",
        F.pmod(
            F.xxhash64(
                F.col("customer_id").cast("string"),
                F.lit("california_city_v1")
            ),
            F.lit(100)
        )
    )
)

network_remapped = network_remapped.withColumn(
    "location_key",

    # Los Angeles Metro
    F.when(
        (F.col("region") == "Los Angeles Metro")
        & (F.col("_u_city") < 38),
        101
    )
    .when(
        (F.col("region") == "Los Angeles Metro")
        & (F.col("_u_city") < 60),
        102
    )
    .when(
        (F.col("region") == "Los Angeles Metro")
        & (F.col("_u_city") < 80),
        103
    )
    .when(
        F.col("region") == "Los Angeles Metro",
        104
    )

    # Bay Area
    .when(
        (F.col("region") == "Bay Area")
        & (F.col("_u_city") < 35),
        201
    )
    .when(
        (F.col("region") == "Bay Area")
        & (F.col("_u_city") < 75),
        202
    )
    .when(
        F.col("region") == "Bay Area",
        203
    )

    # San Diego County
    .when(
        (F.col("region") == "San Diego County")
        & (F.col("_u_city") < 75),
        301
    )
    .when(
        F.col("region") == "San Diego County",
        302
    )

    # Central Valley
    .when(
        (F.col("region") == "Central Valley")
        & (F.col("_u_city") < 34),
        401
    )
    .when(
        (F.col("region") == "Central Valley")
        & (F.col("_u_city") < 67),
        402
    )
    .when(
        F.col("region") == "Central Valley",
        403
    )

    # Sacramento / North
    .when(
        (F.col("region") == "Sacramento / North")
        & (F.col("_u_city") < 75),
        501
    )
    .when(
        F.col("region") == "Sacramento / North",
        502
    )
    .cast("long")
)

# Join the city quality profile
network_remapped = network_remapped.join(
    location_df.select(
        "location_key",
        "quality_factor"
    ),
    on="location_key",
    how="left"
)

# Generate modest synthetic city-level differences
network_remapped = (
    network_remapped

    .withColumn(
        "latency_ms",
        F.round(
            F.col("latency_ms")
            * F.col("quality_factor"),
            2
        )
    )

    .withColumn(
        "packet_loss_pct",
        F.round(
            F.least(
                F.lit(100.0),
                F.col("packet_loss_pct")
                * F.col("quality_factor")
            ),
            2
        )
    )

    .withColumn(
        "download_speed_mbps",
        F.round(
            F.greatest(
                F.lit(0.1),
                F.col("download_speed_mbps")
                / F.col("quality_factor")
            ),
            2
        )
    )

    .withColumn(
        "outage_minutes",
        F.round(
            F.col("outage_minutes")
            * F.col("quality_factor")
        ).cast("long")
    )
)

if "_gold_created_at" in network_base.columns:
    network_remapped = network_remapped.withColumn(
        "_gold_created_at",
        F.current_timestamp()
    )

original_columns = [
    column
    for column in network_base.columns
    if column != "location_key"
]

network_california = network_remapped.select(
    *original_columns,
    "location_key"
)

(
    network_california.write
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

print(
    f"California network events created: "
    f"{network_california.count():,}"
)

# COMMAND ----------

fact_network = spark.table(NETWORK_TABLE)

regional_summary = (
    fact_network
    .groupBy("region")
    .agg(
        F.countDistinct("event_id")
         .alias("network_event_count"),

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

        F.sum("is_outage")
         .alias("outage_event_count"),

        F.sum("is_degraded")
         .alias("degraded_event_count"),

        F.sum("outage_minutes")
         .alias("total_outage_minutes"),

        F.avg(
            F.when(
                F.col("event_type") != "Normal",
                1.0
            ).otherwise(0.0)
        ).alias("_incident_rate")
    )
)

regional_summary = regional_summary.withColumn(
    "regional_quality_score",
    F.round(
        0.40 * (
            100 * (1 - F.col("_incident_rate"))
        )
        + 0.25 * (
            100 * (
                1 - F.least(
                    F.col("average_latency_ms") / 200,
                    F.lit(1.0)
                )
            )
        )
        + 0.20 * (
            100 * (
                1 - F.least(
                    F.col("average_packet_loss_pct") / 10,
                    F.lit(1.0)
                )
            )
        )
        + 0.15 * (
            100 * F.least(
                F.col("average_download_speed_mbps") / 100,
                F.lit(1.0)
            )
        ),
        2
    )
)

regional_gold = (
    regional_summary
    .drop("_incident_rate")
    .withColumn(
        "_gold_created_at",
        F.current_timestamp()
    )
)

(
    regional_gold.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(REGIONAL_TABLE)
)

city_network_base = (
    fact_network.alias("network")
    .join(
        spark.table(LOCATION_TABLE).alias("location"),
        F.col("network.location_key")
        == F.col("location.location_key"),
        how="inner"
    )
    .select(
        F.col("location.city").alias("city"),
        F.col("location.region").alias("region"),
        F.col("location.state").alias("state"),
        F.col("location.country").alias("country"),

        F.col("network.event_id").alias("event_id"),
        F.col("network.customer_id").alias("customer_id"),
        F.col("network.event_type").alias("event_type"),
        F.col("network.latency_ms").alias("latency_ms"),

        F.col(
            "network.download_speed_mbps"
        ).alias("download_speed_mbps"),

        F.col(
            "network.packet_loss_pct"
        ).alias("packet_loss_pct"),

        F.col(
            "network.outage_minutes"
        ).alias("outage_minutes")
    )
)

city_validation = (
    city_network_base
    .groupBy(
        "city",
        "region",
        "state",
        "country"
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
            F.avg("latency_ms"),
            2
        ).alias("average_latency_ms"),

        F.round(
            F.avg("download_speed_mbps"),
            2
        ).alias("average_speed_mbps"),

        F.round(
            F.avg("packet_loss_pct"),
            2
        ).alias("average_packet_loss_pct"),

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
)

display(
    city_validation
    .orderBy(
        F.col("network_incidents").desc()
    )
)