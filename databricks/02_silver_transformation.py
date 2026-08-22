# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG = "workspace"
BRONZE_SCHEMA = "telecom_bronze"
SILVER_SCHEMA = "telecom_silver"

print("Silver transformation environment is ready.")

# COMMAND ----------

customers_bronze = spark.table(
    f"{CATALOG}.{BRONZE_SCHEMA}.customers_raw"
)

print(f"Bronze customer count: {customers_bronze.count():,}")

display(customers_bronze.limit(5))

# COMMAND ----------

customer_window = (
    Window
    .partitionBy("customerID")
    .orderBy(F.col("_ingestion_timestamp").desc())
)

customers_silver = (
    customers_bronze

    # Remove records without a customer identifier
    .filter(
        F.col("customerID").isNotNull()
        & (F.trim(F.col("customerID")) != "")
    )

    # Keep the latest version of every customer
    .withColumn(
        "_row_number",
        F.row_number().over(customer_window)
    )
    .filter(F.col("_row_number") == 1)
    .drop("_row_number")

    # Rename columns using snake_case
    .select(
        F.trim("customerID").alias("customer_id"),
        F.lower(F.trim("gender")).alias("gender"),
        F.col("SeniorCitizen").alias("senior_citizen"),
        F.lower(F.trim("Partner")).alias("has_partner"),
        F.lower(F.trim("Dependents")).alias("has_dependents"),
        F.col("tenure").alias("tenure_months"),
        F.lower(F.trim("PhoneService")).alias("phone_service"),
        F.trim("MultipleLines").alias("multiple_lines"),
        F.trim("InternetService").alias("internet_service"),
        F.trim("OnlineSecurity").alias("online_security"),
        F.trim("OnlineBackup").alias("online_backup"),
        F.trim("DeviceProtection").alias("device_protection"),
        F.trim("TechSupport").alias("tech_support"),
        F.trim("StreamingTV").alias("streaming_tv"),
        F.trim("StreamingMovies").alias("streaming_movies"),
        F.trim("Contract").alias("contract_type"),
        F.lower(F.trim("PaperlessBilling")).alias("paperless_billing"),
        F.trim("PaymentMethod").alias("payment_method"),
        F.col("MonthlyCharges").alias("monthly_charges"),
        F.col("TotalCharges").alias("total_charges"),
        F.trim("Churn").alias("churn")
    )

    # Handle missing TotalCharges
    .withColumn(
        "total_charges",
        F.coalesce(
            F.col("total_charges"),
            F.col("monthly_charges") * F.col("tenure_months")
        )
    )

    # Create a numeric ML target
    .withColumn(
        "churn_label",
        F.when(F.lower(F.col("churn")) == "yes", 1)
        .otherwise(0)
    )

    # Add Silver processing metadata
    .withColumn(
        "_silver_processed_at",
        F.current_timestamp()
    )
)

# COMMAND ----------

CUSTOMERS_SILVER_TABLE = (
    f"{CATALOG}.{SILVER_SCHEMA}.customers_clean"
)

(
    customers_silver.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(CUSTOMERS_SILVER_TABLE)
)

print(f"Created: {CUSTOMERS_SILVER_TABLE}")
print(
    f"Silver customer count: "
    f"{spark.table(CUSTOMERS_SILVER_TABLE).count():,}"
)

# COMMAND ----------

display(
    spark.sql("""
    SELECT
        COUNT(*) AS total_customers,
        COUNT(DISTINCT customer_id) AS unique_customers,
        SUM(CASE WHEN customer_id IS NULL THEN 1 ELSE 0 END)
            AS null_customer_ids,
        SUM(CASE WHEN monthly_charges < 0 THEN 1 ELSE 0 END)
            AS invalid_monthly_charges,
        SUM(CASE WHEN churn_label NOT IN (0, 1) THEN 1 ELSE 0 END)
            AS invalid_churn_labels,
        ROUND(AVG(monthly_charges), 2)
            AS average_monthly_charges,
        ROUND(AVG(tenure_months), 2)
            AS average_tenure
    FROM workspace.telecom_silver.customers_clean
    """)
)

# COMMAND ----------

billing_bronze = spark.table(
    f"{CATALOG}.{BRONZE_SCHEMA}.billing_transactions_raw"
)

billing_window = (
    Window
    .partitionBy("invoice_id")
    .orderBy(F.col("_ingestion_timestamp").desc())
)

billing_silver = (
    billing_bronze

    # Remove records without business keys
    .filter(
        F.col("invoice_id").isNotNull()
        & F.col("customerID").isNotNull()
    )

    # Remove duplicate invoices
    .withColumn(
        "_row_number",
        F.row_number().over(billing_window)
    )
    .filter(F.col("_row_number") == 1)
    .drop("_row_number")

    # Select and standardize columns
    .select(
        F.trim("invoice_id").alias("invoice_id"),
        F.trim("customerID").alias("customer_id"),
        F.to_date("billing_month").alias("billing_month"),
        F.col("MonthlyCharges").alias("monthly_charges"),
        F.col("usage_charges"),
        F.col("discount_amount"),
        F.col("tax_amount"),
        F.col("total_amount"),
        F.trim("PaymentMethod").alias("payment_method"),
        F.lower(F.trim("payment_status")).alias("payment_status"),
        F.to_timestamp("payment_date").alias("payment_date")
    )

    # Reject impossible financial values
    .filter(
        (F.col("monthly_charges") >= 0)
        & (F.col("usage_charges") >= 0)
        & (F.col("discount_amount") >= 0)
        & (F.col("tax_amount") >= 0)
        & (F.col("total_amount") >= 0)
    )

    .withColumn(
        "is_paid",
        F.when(F.col("payment_status") == "paid", 1).otherwise(0)
    )
    .withColumn(
        "is_late",
        F.when(F.col("payment_status") == "late", 1).otherwise(0)
    )
    .withColumn(
        "is_unpaid",
        F.when(F.col("payment_status") == "unpaid", 1).otherwise(0)
    )
    .withColumn("_silver_processed_at", F.current_timestamp())
)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

BILLING_SILVER_TABLE = (
    f"{CATALOG}.{SILVER_SCHEMA}.billing_clean"
)

(
    billing_silver.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(BILLING_SILVER_TABLE)
)

print(
    f"Billing clean count: "
    f"{spark.table(BILLING_SILVER_TABLE).count():,}"
)

# COMMAND ----------

billing_summary = (
    billing_silver
    .groupBy("customer_id")
    .agg(
        F.countDistinct("invoice_id").alias("invoice_count"),
        F.round(F.sum("total_amount"), 2).alias("total_billed_amount"),
        F.round(F.avg("total_amount"), 2).alias("average_invoice_amount"),
        F.round(F.sum("usage_charges"), 2).alias("total_usage_charges"),
        F.round(F.sum("discount_amount"), 2).alias("total_discounts"),
        F.sum("is_paid").alias("paid_invoice_count"),
        F.sum("is_late").alias("late_invoice_count"),
        F.sum("is_unpaid").alias("unpaid_invoice_count"),
        F.max("billing_month").alias("latest_billing_month")
    )
    .withColumn(
        "payment_reliability_score",
        F.round(
            F.col("paid_invoice_count") / F.col("invoice_count"),
            4
        )
    )
)

(
    billing_summary.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(
        f"{CATALOG}.{SILVER_SCHEMA}.customer_billing_summary"
    )
)

display(billing_summary.limit(10))

# COMMAND ----------

complaints_bronze = spark.table(
    f"{CATALOG}.{BRONZE_SCHEMA}.customer_complaints_raw"
)

complaint_window = (
    Window
    .partitionBy("complaint_id")
    .orderBy(F.col("_ingestion_timestamp").desc())
)

complaints_silver = (
    complaints_bronze

    .filter(
        F.col("complaint_id").isNotNull()
        & F.col("customerID").isNotNull()
    )

    .withColumn(
        "_row_number",
        F.row_number().over(complaint_window)
    )
    .filter(F.col("_row_number") == 1)
    .drop("_row_number")

    .select(
        F.trim("complaint_id").alias("complaint_id"),
        F.trim("customerID").alias("customer_id"),
        F.to_timestamp("opened_at").alias("opened_at"),
        F.lower(F.trim("category")).alias("complaint_category"),
        F.lower(F.trim("priority")).alias("priority"),
        F.lower(F.trim("status")).alias("complaint_status"),
        F.lower(F.trim("channel")).alias("complaint_channel"),
        F.col("resolution_hours"),
        F.col("satisfaction_score")
    )

    .filter(
        (F.col("resolution_hours") >= 0)
        & F.col("satisfaction_score").between(1, 5)
    )

    .withColumn(
        "is_critical",
        F.when(F.col("priority") == "critical", 1).otherwise(0)
    )
    .withColumn(
        "is_escalated",
        F.when(F.col("complaint_status") == "escalated", 1).otherwise(0)
    )
    .withColumn(
        "is_unresolved",
        F.when(
            F.col("complaint_status").isin("in progress", "escalated"),
            1
        ).otherwise(0)
    )
    .withColumn("_silver_processed_at", F.current_timestamp())
)

# COMMAND ----------

COMPLAINTS_SILVER_TABLE = (
    f"{CATALOG}.{SILVER_SCHEMA}.complaints_clean"
)

(
    complaints_silver.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(COMPLAINTS_SILVER_TABLE)
)

print(
    f"Complaints clean count: "
    f"{spark.table(COMPLAINTS_SILVER_TABLE).count():,}"
)

# COMMAND ----------

complaints_summary = (
    complaints_silver
    .groupBy("customer_id")
    .agg(
        F.countDistinct("complaint_id").alias("complaint_count"),
        F.round(
            F.avg("resolution_hours"), 2
        ).alias("average_resolution_hours"),
        F.round(
            F.avg("satisfaction_score"), 2
        ).alias("average_satisfaction_score"),
        F.sum("is_critical").alias("critical_complaint_count"),
        F.sum("is_escalated").alias("escalated_complaint_count"),
        F.sum("is_unresolved").alias("unresolved_complaint_count"),
        F.max("opened_at").alias("latest_complaint_at")
    )
)

(
    complaints_summary.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(
        f"{CATALOG}.{SILVER_SCHEMA}.customer_complaints_summary"
    )
)

display(complaints_summary.limit(10))

# COMMAND ----------

network_bronze = spark.table(
    f"{CATALOG}.{BRONZE_SCHEMA}.network_events_raw"
)

network_window = (
    Window
    .partitionBy("event_id")
    .orderBy(F.col("_ingestion_timestamp").desc())
)

network_silver = (
    network_bronze

    .filter(
        F.col("event_id").isNotNull()
        & F.col("customerID").isNotNull()
    )

    .withColumn(
        "_row_number",
        F.row_number().over(network_window)
    )
    .filter(F.col("_row_number") == 1)
    .drop("_row_number")

    .select(
        F.trim("event_id").alias("event_id"),
        F.trim("customerID").alias("customer_id"),
        F.to_timestamp("event_timestamp").alias("event_timestamp"),
        F.lower(F.trim("event_type")).alias("event_type"),
        F.trim("region").alias("region"),
        F.trim("tower_id").alias("tower_id"),
        F.col("latency_ms"),
        F.col("packet_loss_pct"),
        F.col("download_speed_mbps"),
        F.col("outage_minutes")
    )

    .filter(
        (F.col("latency_ms") >= 0)
        & F.col("packet_loss_pct").between(0, 100)
        & (F.col("download_speed_mbps") >= 0)
        & (F.col("outage_minutes") >= 0)
    )

    .withColumn(
        "is_outage",
        F.when(F.col("event_type") == "outage", 1).otherwise(0)
    )
    .withColumn(
        "is_degraded",
        F.when(
            F.col("event_type").isin(
                "high latency",
                "packet loss",
                "service degradation",
                "outage"
            ),
            1
        ).otherwise(0)
    )
    .withColumn("_silver_processed_at", F.current_timestamp())
)

# COMMAND ----------

NETWORK_SILVER_TABLE = (
    f"{CATALOG}.{SILVER_SCHEMA}.network_events_clean"
)

(
    network_silver.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(NETWORK_SILVER_TABLE)
)

print(
    f"Network clean count: "
    f"{spark.table(NETWORK_SILVER_TABLE).count():,}"
)


# COMMAND ----------

network_summary = (
    network_silver
    .groupBy("customer_id")
    .agg(
        F.countDistinct("event_id").alias("network_event_count"),
        F.round(F.avg("latency_ms"), 2).alias("average_latency_ms"),
        F.round(
            F.avg("packet_loss_pct"), 2
        ).alias("average_packet_loss_pct"),
        F.round(
            F.avg("download_speed_mbps"), 2
        ).alias("average_download_speed_mbps"),
        F.sum("outage_minutes").alias("total_outage_minutes"),
        F.sum("is_outage").alias("outage_event_count"),
        F.sum("is_degraded").alias("degraded_event_count"),
        F.countDistinct("tower_id").alias("distinct_tower_count"),
        F.max("event_timestamp").alias("latest_network_event_at")
    )
    .withColumn(
        "network_quality_score",
        F.round(
            F.greatest(
                F.lit(0.0),
                F.lit(100.0)
                - F.col("average_latency_ms") * 0.10
                - F.col("average_packet_loss_pct") * 1.50
                - F.col("outage_event_count") * 2.00
            ),
            2
        )
    )
)

(
    network_summary.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(
        f"{CATALOG}.{SILVER_SCHEMA}.customer_network_summary"
    )
)

display(network_summary.limit(10))


# COMMAND ----------

silver_summary = spark.sql("""
SELECT 'customers_clean' AS table_name, COUNT(*) AS record_count
FROM workspace.telecom_silver.customers_clean

UNION ALL

SELECT 'billing_clean', COUNT(*)
FROM workspace.telecom_silver.billing_clean

UNION ALL

SELECT 'customer_billing_summary', COUNT(*)
FROM workspace.telecom_silver.customer_billing_summary

UNION ALL

SELECT 'complaints_clean', COUNT(*)
FROM workspace.telecom_silver.complaints_clean

UNION ALL

SELECT 'customer_complaints_summary', COUNT(*)
FROM workspace.telecom_silver.customer_complaints_summary

UNION ALL

SELECT 'network_events_clean', COUNT(*)
FROM workspace.telecom_silver.network_events_clean

UNION ALL

SELECT 'customer_network_summary', COUNT(*)
FROM workspace.telecom_silver.customer_network_summary
""")

display(silver_summary)

# COMMAND ----------

customers_df = spark.table(
    "workspace.telecom_silver.customers_clean"
)

billing_df = spark.table(
    "workspace.telecom_silver.customer_billing_summary"
)

complaints_df = spark.table(
    "workspace.telecom_silver.customer_complaints_summary"
)

network_df = spark.table(
    "workspace.telecom_silver.customer_network_summary"
)

print("All Silver sources loaded successfully.")

# COMMAND ----------

customer_360 = (
    customers_df

    .join(
        billing_df,
        on="customer_id",
        how="left"
    )

    .join(
        complaints_df,
        on="customer_id",
        how="left"
    )

    .join(
        network_df,
        on="customer_id",
        how="left"
    )
)

# COMMAND ----------

customer_360 = (
    customer_360
    .withColumn(
        "has_billing_history",
        F.when(F.col("invoice_count").isNotNull(), 1).otherwise(0)
    )
    .withColumn(
        "has_complaints",
        F.when(F.col("complaint_count").isNotNull(), 1).otherwise(0)
    )
    .withColumn(
        "has_network_events",
        F.when(F.col("network_event_count").isNotNull(), 1).otherwise(0)
    )
)

# COMMAND ----------

zero_fill_columns = [
    "invoice_count",
    "total_billed_amount",
    "average_invoice_amount",
    "total_usage_charges",
    "total_discounts",
    "paid_invoice_count",
    "late_invoice_count",
    "unpaid_invoice_count",
    "payment_reliability_score",
    "complaint_count",
    "average_resolution_hours",
    "average_satisfaction_score",
    "critical_complaint_count",
    "escalated_complaint_count",
    "unresolved_complaint_count",
    "network_event_count",
    "average_latency_ms",
    "average_packet_loss_pct",
    "average_download_speed_mbps",
    "total_outage_minutes",
    "outage_event_count",
    "degraded_event_count",
    "distinct_tower_count",
    "network_quality_score"
]

customer_360 = customer_360.fillna(
    0,
    subset=zero_fill_columns
)

# COMMAND ----------

customer_360 = (
    customer_360

    .withColumn(
        "payment_issue_count",
        F.col("late_invoice_count")
        + F.col("unpaid_invoice_count")
    )

    .withColumn(
        "complaint_severity_score",
        F.col("complaint_count")
        + F.col("critical_complaint_count") * 2
        + F.col("escalated_complaint_count") * 2
        + F.col("unresolved_complaint_count")
    )

    .withColumn(
        "network_issue_rate",
        F.when(
            F.col("network_event_count") > 0,
            F.round(
                F.col("degraded_event_count")
                / F.col("network_event_count"),
                4
            )
        ).otherwise(0.0)
    )

    .withColumn(
        "revenue_at_risk",
        F.when(
            F.col("churn_label") == 1,
            F.col("monthly_charges")
        ).otherwise(0.0)
    )

    .withColumn(
        "_customer_360_created_at",
        F.current_timestamp()
    )
)

# COMMAND ----------

CUSTOMER_360_TABLE = (
    "workspace.telecom_silver.customer_360"
)

(
    customer_360.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(CUSTOMER_360_TABLE)
)

print(f"Created: {CUSTOMER_360_TABLE}")
print(
    f"Customer count: "
    f"{spark.table(CUSTOMER_360_TABLE).count():,}"
)

# COMMAND ----------

display(
    spark.sql("""
    SELECT
        COUNT(*) AS total_rows,
        COUNT(DISTINCT customer_id) AS unique_customers,

        SUM(
            CASE WHEN customer_id IS NULL
            THEN 1 ELSE 0 END
        ) AS null_customer_ids,

        SUM(has_billing_history) AS customers_with_billing,
        SUM(has_complaints) AS customers_with_complaints,
        SUM(has_network_events) AS customers_with_network_events,

        ROUND(AVG(complaint_count), 2)
            AS average_complaints,

        ROUND(AVG(network_quality_score), 2)
            AS average_network_quality,

        ROUND(SUM(revenue_at_risk), 2)
            AS known_revenue_at_risk

    FROM workspace.telecom_silver.customer_360
    """)
)

# COMMAND ----------

