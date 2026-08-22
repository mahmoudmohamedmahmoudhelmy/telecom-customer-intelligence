# Databricks notebook source
from pyspark.sql import functions as F

CATALOG = "workspace"
SILVER_SCHEMA = "telecom_silver"
GOLD_SCHEMA = "telecom_gold"

print("Gold Analytics environment is ready.")

# COMMAND ----------

customers_clean = spark.table(
    f"{CATALOG}.{SILVER_SCHEMA}.customers_clean"
)

dim_customer = (
    customers_clean
    .select(
        F.xxhash64("customer_id").alias("customer_key"),
        "customer_id",
        "gender",
        "senior_citizen",
        "has_partner",
        "has_dependents",
        "tenure_months",
        "phone_service",
        "multiple_lines",
        "internet_service",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
        "contract_type",
        "paperless_billing",
        "payment_method",
        "monthly_charges",
        "total_charges",
        "churn",
        "churn_label"
    )
    .withColumn(
        "tenure_segment",
        F.when(F.col("tenure_months") <= 12, "0-12 Months")
        .when(F.col("tenure_months") <= 24, "13-24 Months")
        .when(F.col("tenure_months") <= 48, "25-48 Months")
        .otherwise("49+ Months")
    )
    .withColumn(
        "monthly_charge_segment",
        F.when(F.col("monthly_charges") < 40, "Low Value")
        .when(F.col("monthly_charges") < 80, "Medium Value")
        .otherwise("High Value")
    )
    .withColumn("_gold_created_at", F.current_timestamp())
)

(
    dim_customer.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(
        f"{CATALOG}.{GOLD_SCHEMA}.dim_customer"
    )
)

print(
    "dim_customer count:",
    spark.table(
        f"{CATALOG}.{GOLD_SCHEMA}.dim_customer"
    ).count()
)

# COMMAND ----------

date_range = spark.sql("""
SELECT EXPLODE(
    SEQUENCE(
        TO_DATE('2025-01-01'),
        TO_DATE('2026-12-31'),
        INTERVAL 1 DAY
    )
) AS full_date
""")

dim_date = (
    date_range
    .withColumn(
        "date_key",
        F.date_format("full_date", "yyyyMMdd").cast("int")
    )
    .withColumn("year", F.year("full_date"))
    .withColumn("quarter", F.quarter("full_date"))
    .withColumn("month_number", F.month("full_date"))
    .withColumn("month_name", F.date_format("full_date", "MMMM"))
    .withColumn("year_month", F.date_format("full_date", "yyyy-MM"))
    .withColumn("week_number", F.weekofyear("full_date"))
    .withColumn("day_of_month", F.dayofmonth("full_date"))
    .withColumn("day_name", F.date_format("full_date", "EEEE"))
    .withColumn(
        "is_weekend",
        F.when(
            F.dayofweek("full_date").isin(1, 7),
            1
        ).otherwise(0)
    )
    .select(
        "date_key",
        "full_date",
        "year",
        "quarter",
        "month_number",
        "month_name",
        "year_month",
        "week_number",
        "day_of_month",
        "day_name",
        "is_weekend"
    )
)

(
    dim_date.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(
        f"{CATALOG}.{GOLD_SCHEMA}.dim_date"
    )
)

display(dim_date.limit(10))

# COMMAND ----------

display(
    spark.sql("""
    SELECT
        'dim_customer' AS table_name,
        COUNT(*) AS row_count
    FROM workspace.telecom_gold.dim_customer

    UNION ALL

    SELECT
        'dim_date',
        COUNT(*)
    FROM workspace.telecom_gold.dim_date
    """)
)

# COMMAND ----------

customer_keys = (
    spark.table("workspace.telecom_gold.dim_customer")
    .select(
        "customer_key",
        "customer_id"
    )
)

# COMMAND ----------

billing_clean = spark.table(
    "workspace.telecom_silver.billing_clean"
)

fact_billing = (
    billing_clean
    .join(
        customer_keys,
        on="customer_id",
        how="left"
    )
    .withColumn(
        "billing_date_key",
        F.date_format("billing_month", "yyyyMMdd").cast("int")
    )
    .withColumn(
        "payment_date_key",
        F.date_format("payment_date", "yyyyMMdd").cast("int")
    )
    .select(
        "invoice_id",
        "customer_key",
        "customer_id",
        "billing_date_key",
        "payment_date_key",
        "billing_month",
        "payment_date",
        "monthly_charges",
        "usage_charges",
        "discount_amount",
        "tax_amount",
        "total_amount",
        "payment_method",
        "payment_status",
        "is_paid",
        "is_late",
        "is_unpaid"
    )
    .withColumn("_gold_created_at", F.current_timestamp())
)

(
    fact_billing.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.telecom_gold.fact_billing")
)

print(f"Fact Billing: {fact_billing.count():,}")

# COMMAND ----------

complaints_clean = spark.table(
    "workspace.telecom_silver.complaints_clean"
)

fact_complaints = (
    complaints_clean
    .join(
        customer_keys,
        on="customer_id",
        how="left"
    )
    .withColumn(
        "opened_date_key",
        F.date_format("opened_at", "yyyyMMdd").cast("int")
    )
    .select(
        "complaint_id",
        "customer_key",
        "customer_id",
        "opened_date_key",
        "opened_at",
        "complaint_category",
        "priority",
        "complaint_status",
        "complaint_channel",
        "resolution_hours",
        "satisfaction_score",
        "is_critical",
        "is_escalated",
        "is_unresolved"
    )
    .withColumn("_gold_created_at", F.current_timestamp())
)

(
    fact_complaints.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.telecom_gold.fact_complaints")
)

print(f"Fact Complaints: {fact_complaints.count():,}")

# COMMAND ----------

network_clean = spark.table(
    "workspace.telecom_silver.network_events_clean"
)

fact_network_events = (
    network_clean
    .join(
        customer_keys,
        on="customer_id",
        how="left"
    )
    .withColumn(
        "event_date_key",
        F.date_format("event_timestamp", "yyyyMMdd").cast("int")
    )
    .select(
        "event_id",
        "customer_key",
        "customer_id",
        "event_date_key",
        "event_timestamp",
        "event_type",
        "region",
        "tower_id",
        "latency_ms",
        "packet_loss_pct",
        "download_speed_mbps",
        "outage_minutes",
        "is_outage",
        "is_degraded"
    )
    .withColumn("_gold_created_at", F.current_timestamp())
)

(
    fact_network_events.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.telecom_gold.fact_network_events")
)

print(f"Fact Network Events: {fact_network_events.count():,}")

# COMMAND ----------

display(
    spark.sql("""
    SELECT
        'fact_billing' AS table_name,
        COUNT(*) AS total_rows,
        SUM(
            CASE WHEN customer_key IS NULL
            THEN 1 ELSE 0 END
        ) AS orphan_records
    FROM workspace.telecom_gold.fact_billing

    UNION ALL

    SELECT
        'fact_complaints',
        COUNT(*),
        SUM(
            CASE WHEN customer_key IS NULL
            THEN 1 ELSE 0 END
        )
    FROM workspace.telecom_gold.fact_complaints

    UNION ALL

    SELECT
        'fact_network_events',
        COUNT(*),
        SUM(
            CASE WHEN customer_key IS NULL
            THEN 1 ELSE 0 END
        )
    FROM workspace.telecom_gold.fact_network_events
    """)
)

# COMMAND ----------

customer_360 = spark.table(
    "workspace.telecom_silver.customer_360"
)

customer_churn_analytics = (
    customer_360

    # Business risk score without using the actual churn label
    .withColumn(
        "risk_score",
        F.least(
            F.lit(100.0),

            F.when(
                F.lower(F.col("contract_type")) == "month-to-month",
                20.0
            ).otherwise(5.0)

            + F.when(
                F.col("tenure_months") <= 12,
                15.0
            ).when(
                F.col("tenure_months") <= 24,
                10.0
            ).otherwise(3.0)

            + F.least(
                F.col("payment_issue_count") * 8.0,
                F.lit(20.0)
            )

            + F.least(
                F.col("complaint_severity_score") * 3.0,
                F.lit(20.0)
            )

            + F.when(
                F.col("has_network_events") == 1,
                F.least(
                    (100.0 - F.col("network_quality_score")) * 0.25,
                    F.lit(25.0)
                )
            ).otherwise(0.0)
        )
    )

    .withColumn(
        "risk_category",
        F.when(F.col("risk_score") >= 70, "Critical")
        .when(F.col("risk_score") >= 50, "High")
        .when(F.col("risk_score") >= 30, "Medium")
        .otherwise("Low")
    )

    .withColumn(
        "customer_value_segment",
        F.when(F.col("monthly_charges") >= 80, "High Value")
        .when(F.col("monthly_charges") >= 40, "Medium Value")
        .otherwise("Low Value")
    )

    .withColumn(
        "actual_churn_revenue",
        F.when(
            F.col("churn_label") == 1,
            F.col("monthly_charges")
        ).otherwise(0.0)
    )

    .withColumn(
        "business_risk_revenue",
        F.when(
            F.col("risk_category").isin("High", "Critical"),
            F.col("monthly_charges")
        ).otherwise(0.0)
    )

    .withColumn(
        "recommended_action",
        F.when(
            F.col("unpaid_invoice_count") > 0,
            "Payment support and retention offer"
        )
        .when(
            F.col("unresolved_complaint_count") > 0,
            "Resolve open complaints immediately"
        )
        .when(
            F.col("network_quality_score") < 60,
            "Network investigation and service credit"
        )
        .when(
            F.lower(F.col("contract_type")) == "month-to-month",
            "Offer long-term contract discount"
        )
        .otherwise("Standard engagement")
    )

    .withColumn("_gold_created_at", F.current_timestamp())
)

(
    customer_churn_analytics.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(
        "workspace.telecom_gold.customer_churn_analytics"
    )
)

print(
    "Customer analytics count:",
    customer_churn_analytics.count()
)

# COMMAND ----------

executive_kpis = (
    customer_churn_analytics
    .agg(
        F.countDistinct("customer_id").alias("total_customers"),

        F.sum("churn_label").alias("churned_customers"),

        F.round(
            F.avg("churn_label") * 100,
            2
        ).alias("churn_rate_pct"),

        F.round(
            F.sum("monthly_charges"),
            2
        ).alias("monthly_recurring_revenue"),

        F.round(
            F.sum("actual_churn_revenue"),
            2
        ).alias("actual_churn_revenue"),

        F.round(
            F.sum("business_risk_revenue"),
            2
        ).alias("business_risk_revenue"),

        F.round(
            F.avg("monthly_charges"),
            2
        ).alias("average_revenue_per_user"),

        F.sum("complaint_count").alias("total_complaints"),

        F.sum("unresolved_complaint_count")
        .alias("unresolved_complaints"),

        F.sum("outage_event_count").alias("total_outage_events"),

        F.round(
            F.avg(
                F.when(
                    F.col("has_network_events") == 1,
                    F.col("network_quality_score")
                )
            ),
            2
        ).alias("average_network_quality_score"),

        F.sum(
            F.when(
                F.col("risk_category").isin("High", "Critical"),
                1
            ).otherwise(0)
        ).alias("high_risk_customers")
    )

    .withColumn("snapshot_date", F.current_date())
    .withColumn("_gold_created_at", F.current_timestamp())
)

(
    executive_kpis.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(
        "workspace.telecom_gold.executive_kpis"
    )
)

display(executive_kpis)

# COMMAND ----------

revenue_risk_by_segment = (
    customer_churn_analytics

    .groupBy(
        "contract_type",
        "internet_service",
        "customer_value_segment"
    )

    .agg(
        F.countDistinct("customer_id").alias("customer_count"),

        F.sum("churn_label").alias("churned_customers"),

        F.round(
            F.avg("churn_label") * 100,
            2
        ).alias("churn_rate_pct"),

        F.round(
            F.sum("monthly_charges"),
            2
        ).alias("monthly_revenue"),

        F.round(
            F.sum("actual_churn_revenue"),
            2
        ).alias("actual_churn_revenue"),

        F.round(
            F.sum("business_risk_revenue"),
            2
        ).alias("business_risk_revenue"),

        F.round(
            F.avg("risk_score"),
            2
        ).alias("average_risk_score")
    )

    .withColumn("_gold_created_at", F.current_timestamp())
)

(
    revenue_risk_by_segment.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(
        "workspace.telecom_gold.revenue_risk_by_segment"
    )
)

display(
    revenue_risk_by_segment
    .orderBy(F.col("churn_rate_pct").desc())
)

# COMMAND ----------

network_performance_by_region = (
    spark.table("workspace.telecom_gold.fact_network_events")

    .groupBy("region")

    .agg(
        F.countDistinct("event_id").alias("network_event_count"),

        F.countDistinct("customer_id").alias("affected_customers"),

        F.round(
            F.avg("latency_ms"),
            2
        ).alias("average_latency_ms"),

        F.round(
            F.avg("packet_loss_pct"),
            2
        ).alias("average_packet_loss_pct"),

        F.round(
            F.avg("download_speed_mbps"),
            2
        ).alias("average_download_speed_mbps"),

        F.sum("is_outage").alias("outage_event_count"),

        F.sum("is_degraded").alias("degraded_event_count"),

        F.sum("outage_minutes").alias("total_outage_minutes")
    )

    .withColumn(
        "regional_quality_score",
        F.round(
            F.greatest(
                F.lit(0.0),
                F.lit(100.0)
                - F.col("average_latency_ms") * 0.10
                - F.col("average_packet_loss_pct") * 1.50
                - F.col("outage_event_count") * 0.01
            ),
            2
        )
    )

    .withColumn("_gold_created_at", F.current_timestamp())
)

(
    network_performance_by_region.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(
        "workspace.telecom_gold.network_performance_by_region"
    )
)

display(network_performance_by_region)

# COMMAND ----------

customer_ml_features = (
    customer_churn_analytics

    .select(
        "customer_id",

        # Customer characteristics
        "gender",
        "senior_citizen",
        "has_partner",
        "has_dependents",
        "tenure_months",

        # Services
        "phone_service",
        "internet_service",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",

        # Contract and financial features
        "contract_type",
        "paperless_billing",
        "payment_method",
        "monthly_charges",
        "total_charges",
        "invoice_count",
        "average_invoice_amount",
        "payment_issue_count",
        "payment_reliability_score",

        # Customer-experience features
        "complaint_count",
        "critical_complaint_count",
        "escalated_complaint_count",
        "unresolved_complaint_count",
        "average_resolution_hours",
        "average_satisfaction_score",

        # Network features
        "network_event_count",
        "average_latency_ms",
        "average_packet_loss_pct",
        "average_download_speed_mbps",
        "total_outage_minutes",
        "outage_event_count",
        "degraded_event_count",
        "network_quality_score",

        # Target
        "churn_label"
    )

    .withColumn("_feature_created_at", F.current_timestamp())
)

(
    customer_ml_features.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(
        "workspace.telecom_gold.customer_ml_features"
    )
)

print(
    "ML Feature count:",
    customer_ml_features.count()
)

# COMMAND ----------

display(
    spark.sql("""
    SHOW TABLES IN workspace.telecom_gold
    """)
)