# Databricks notebook source
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


current_user = spark.sql(
    "SELECT current_user()"
).first()[0]

experiment_path = (
    f"/Users/{current_user}/telecom_churn_prediction"
)

mlflow.set_experiment(experiment_path)

print(f"MLflow experiment: {experiment_path}")

# COMMAND ----------

features_spark_df = spark.table(
    "workspace.telecom_gold.customer_ml_features"
)

print(f"Dataset size: {features_spark_df.count():,}")

display(
    features_spark_df
    .groupBy("churn_label")
    .count()
)

# COMMAND ----------

features_pdf = (
    features_spark_df
    .drop("_feature_created_at")
    .toPandas()
)

customer_ids = features_pdf["customer_id"].copy()

X = features_pdf.drop(
    columns=["customer_id", "churn_label"]
)

y = features_pdf["churn_label"]

print("Feature shape:", X.shape)
print("Target distribution:")
print(y.value_counts(normalize=True))

# COMMAND ----------

categorical_columns = [
    "gender",
    "has_partner",
    "has_dependents",
    "phone_service",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "contract_type",
    "paperless_billing",
    "payment_method"
]

numeric_columns = [
    column
    for column in X.columns
    if column not in categorical_columns
]

print("Categorical features:", len(categorical_columns))
print("Numeric features:", len(numeric_columns))

# COMMAND ----------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ]
)

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# COMMAND ----------

def train_and_log_model(
    model_name,
    classifier,
    parameters
):
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    with mlflow.start_run(
        run_name=model_name
    ) as run:

        model_pipeline.fit(
            X_train,
            y_train
        )

        predictions = model_pipeline.predict(X_test)

        probabilities = model_pipeline.predict_proba(
            X_test
        )[:, 1]

        metrics = {
            "accuracy": accuracy_score(
                y_test,
                predictions
            ),
            "precision": precision_score(
                y_test,
                predictions,
                zero_division=0
            ),
            "recall": recall_score(
                y_test,
                predictions,
                zero_division=0
            ),
            "f1_score": f1_score(
                y_test,
                predictions,
                zero_division=0
            ),
            "roc_auc": roc_auc_score(
                y_test,
                probabilities
            )
        }

        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)

        signature = infer_signature(
            X_train.head(10),
            model_pipeline.predict(
                X_train.head(10)
            )
        )

        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(5)
        )

        print(f"\nModel: {model_name}")
        print(f"Run ID: {run.info.run_id}")
        print(metrics)

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        )

        print("Confusion Matrix:")
        print(
            confusion_matrix(
                y_test,
                predictions
            )
        )

        return {
            "model_name": model_name,
            "model": model_pipeline,
            "run_id": run.info.run_id,
            **metrics
        }

# COMMAND ----------

logistic_result = train_and_log_model(
    model_name="Logistic Regression",

    classifier=LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    parameters={
        "algorithm": "LogisticRegression",
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": 42
    }
)

# COMMAND ----------

random_forest_result = train_and_log_model(
    model_name="Random Forest",

    classifier=RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    parameters={
        "algorithm": "RandomForestClassifier",
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_split": 5,
        "class_weight": "balanced",
        "random_state": 42
    }
)

# COMMAND ----------

all_results = [
    logistic_result,
    random_forest_result
]

best_result = max(
    all_results,
    key=lambda result: result["roc_auc"]
)

best_model = best_result["model"]

print("Best model:", best_result["model_name"])
print("ROC-AUC:", round(best_result["roc_auc"], 4))
print("F1 Score:", round(best_result["f1_score"], 4))

# COMMAND ----------

model_results = pd.DataFrame([
    {
        key: value
        for key, value in logistic_result.items()
        if key != "model"
    },
    {
        key: value
        for key, value in random_forest_result.items()
        if key != "model"
    }
])

model_results = model_results.sort_values(
    by="roc_auc",
    ascending=False
)

display(model_results)

# COMMAND ----------

all_results = [
    logistic_result,
    random_forest_result
]

best_result = max(
    all_results,
    key=lambda result: result["roc_auc"]
)

best_model = best_result["model"]

print("Best model:", best_result["model_name"])
print("ROC-AUC:", round(best_result["roc_auc"], 4))
print("F1 Score:", round(best_result["f1_score"], 4))
print("MLflow Run ID:", best_result["run_id"])

# COMMAND ----------

all_predictions = best_model.predict(X)

all_probabilities = best_model.predict_proba(X)[:, 1]

predictions_pdf = pd.DataFrame({
    "customer_id": customer_ids,
    "actual_churn_label": y,
    "predicted_churn_label": all_predictions,
    "churn_probability": all_probabilities
})

predictions_pdf["risk_category"] = pd.cut(
    predictions_pdf["churn_probability"],
    bins=[-0.01, 0.30, 0.50, 0.70, 1.00],
    labels=[
        "Low",
        "Medium",
        "High",
        "Critical"
    ]
)

predictions_pdf["model_name"] = best_result["model_name"]
predictions_pdf["mlflow_run_id"] = best_result["run_id"]

predictions_pdf.head(10)

# COMMAND ----------

from pyspark.sql import functions as F

predictions_spark = spark.createDataFrame(
    predictions_pdf
)

predictions_spark = (
    predictions_spark
    .withColumn(
        "churn_probability",
        F.round(F.col("churn_probability"), 4)
    )
    .withColumn(
        "prediction_timestamp",
        F.current_timestamp()
    )
)

display(
    predictions_spark
    .orderBy(
        F.col("churn_probability").desc()
    )
    .limit(10)
)

# COMMAND ----------

(
    predictions_spark.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(
        "workspace.telecom_gold.customer_churn_predictions"
    )
)

print(
    "Prediction count:",
    spark.table(
        "workspace.telecom_gold.customer_churn_predictions"
    ).count()
)

# COMMAND ----------

customer_analytics = spark.table(
    "workspace.telecom_gold.customer_churn_analytics"
)

prediction_results = spark.table(
    "workspace.telecom_gold.customer_churn_predictions"
)

high_risk_customers = (
    customer_analytics.alias("customer")

    .join(
        prediction_results.alias("prediction"),
        on="customer_id",
        how="inner"
    )

    .select(
        "customer_id",
        "contract_type",
        "internet_service",
        "payment_method",
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "payment_issue_count",
        "complaint_count",
        "unresolved_complaint_count",
        "average_satisfaction_score",
        "average_latency_ms",
        "average_packet_loss_pct",
        "total_outage_minutes",
        "network_quality_score",

        F.col(
            "prediction.predicted_churn_label"
        ).alias("predicted_churn"),

        F.col(
            "prediction.churn_probability"
        ).alias("churn_probability"),

        F.col(
            "prediction.risk_category"
        ).alias("ml_risk_category"),

        F.col(
            "customer.recommended_action"
        ).alias("recommended_action")
    )

    .withColumn(
        "predicted_revenue_at_risk",
        F.when(
            F.col("predicted_churn") == 1,
            F.col("monthly_charges")
        ).otherwise(0.0)
    )

    .withColumn(
        "retention_priority",
        F.when(
            (F.col("churn_probability") >= 0.70)
            & (F.col("monthly_charges") >= 80),
            "P1 - Immediate"
        )
        .when(
            F.col("churn_probability") >= 0.70,
            "P2 - Urgent"
        )
        .when(
            F.col("churn_probability") >= 0.50,
            "P3 - Monitor"
        )
        .otherwise("P4 - Standard")
    )

    .withColumn(
        "_gold_created_at",
        F.current_timestamp()
    )
)

(
    high_risk_customers.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(
        "workspace.telecom_gold.high_risk_customers"
    )
)

display(
    high_risk_customers
    .orderBy(
        F.col("churn_probability").desc(),
        F.col("monthly_charges").desc()
    )
    .limit(20)
)

# COMMAND ----------

model_results_for_spark = model_results.copy()

model_results_for_spark["evaluation_timestamp"] = (
    pd.Timestamp.now()
)

model_performance_spark = spark.createDataFrame(
    model_results_for_spark
)

(
    model_performance_spark.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(
        "workspace.telecom_gold.ml_model_performance"
    )
)

display(model_performance_spark)

# COMMAND ----------

trained_preprocessor = best_model.named_steps["preprocessor"]
trained_classifier = best_model.named_steps["classifier"]

feature_names = (
    trained_preprocessor
    .get_feature_names_out()
)

if hasattr(trained_classifier, "feature_importances_"):
    importance_values = trained_classifier.feature_importances_

elif hasattr(trained_classifier, "coef_"):
    importance_values = abs(
        trained_classifier.coef_[0]
    )

else:
    importance_values = None


if importance_values is not None:
    feature_importance_pdf = pd.DataFrame({
        "feature_name": feature_names,
        "importance": importance_values
    })

    feature_importance_pdf = (
        feature_importance_pdf
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    feature_importance_spark = spark.createDataFrame(
        feature_importance_pdf
    )

    (
        feature_importance_spark.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(
            "workspace.telecom_gold.ml_feature_importance"
        )
    )

    display(feature_importance_spark.limit(20))

# COMMAND ----------

from pyspark.sql import functions as F

prediction_validation = (
    spark.table(
        "workspace.telecom_gold.customer_churn_predictions"
    )
    .groupBy(
        "actual_churn_label",
        "predicted_churn_label"
    )
    .agg(
        F.countDistinct("customer_id").alias("customers")
    )
    .orderBy(
        "actual_churn_label",
        "predicted_churn_label"
    )
)

display(prediction_validation)

# COMMAND ----------

