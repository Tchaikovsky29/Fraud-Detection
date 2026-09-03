import os
import time
from src.configuration.lakefs_connection import fetch_batch
import mlflow
from mlflow.tracking import MlflowClient
from prometheus_client import Gauge, start_http_server

DRIFT_CHECK_INTERVAL_SECONDS = int(os.environ.get("DRIFT_CHECK_INTERVAL_SECONDS", 300))
CURRENT_DATA_BATCH = fetch_batch()[-1]

DATASET_DRIFT_DETECTED = Gauge(
    "drift_dataset_drift_detected", "1 if Evidently flagged overall dataset drift, else 0"
)
DRIFT_SHARE = Gauge(
    "drift_share_of_drifted_columns", "Share of columns Evidently flagged as drifted"
)
COLUMN_DRIFT_SCORE = Gauge(
    "drift_column_score", "Per-column drift score", ["column"]
)
LAST_CHECK_SUCCESS = Gauge(
    "drift_last_check_success", "1 if the most recent drift check completed without error"
)


def resolve_champion_reference_batches(client, registered_model_name):
    champion_version = client.get_model_version_by_alias(registered_model_name, "champion")
    champion_run = client.get_run(champion_version.run_id)
    parent_run_id = champion_run.data.tags["mlflow.parentRunId"]
    parent_run = client.get_run(parent_run_id)
    data_used = parent_run.data.params["data_used"]
    return [b.strip() for b in data_used.split(" + ")]


def load_cleaned_batch(spark, batch_paths, repo_name):
    """Reads raw batch(es) and applies the same cleaning as data_cleaning_component,
    so drift is measured on human-readable columns, not opaque vector indices."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import IntegerType, FloatType

    full_paths = [f"s3a://{repo_name}/main/{p}" for p in batch_paths]
    df = spark.read.parquet(*full_paths)

    df = df.withColumn("Transaction Date", F.to_timestamp("Transaction Date"))
    df = df.withColumn("Transaction Day", F.dayofmonth("Transaction Date"))
    df = df.withColumn("Transaction DOW", (F.dayofweek("Transaction Date") + 5) % 7)
    df = df.withColumn("Transaction Month", F.month("Transaction Date"))
    mean_row = df.select(F.round(F.mean("Customer Age"), 0).alias("m")).collect()[0]
    mean_age = float(mean_row["m"])
    df = df.withColumn(
        "Customer Age",
        F.when(F.col("Customer Age") <= -9, F.abs(F.col("Customer Age"))).otherwise(F.col("Customer Age")),
    )
    df = df.withColumn(
        "Customer Age",
        F.when(F.col("Customer Age") < 9, F.lit(mean_age)).otherwise(F.col("Customer Age")),
    )
    df = df.withColumn(
        "Is Address Match", (F.col("Shipping Address") == F.col("Billing Address")).cast("int")
    )
    drop_cols = ["Transaction ID", "Customer ID", "Customer Location", "IP Address",
                 "Transaction Date", "Shipping Address", "Billing Address"]
    df = df.drop(*[c for c in drop_cols if c in df.columns])
    for field in df.schema.fields:
        type_name = field.dataType.typeName()
        if type_name in ("long", "short", "byte"):
            df = df.withColumn(field.name, F.col(field.name).cast(IntegerType()))
        elif type_name == "double":
            df = df.withColumn(field.name, F.col(field.name).cast(FloatType()))

    return df.toPandas()


def run_drift_check():
    from src.utils.main_utils import get_spark_session
    import dagshub

    dagshub.auth.add_app_token(os.environ["DAGSHUB_USER_TOKEN"])
    dagshub.init(
        repo_owner=os.environ["DAGSHUB_REPO_OWNER"], repo_name=os.environ["DAGSHUB_REPO_NAME"], mlflow=True
    )
    client = MlflowClient()
    registered_model_name = "fraud_detection_model"
    repo_name = os.environ["LAKEFS_REPO_NAME"]

    spark = get_spark_session("drift-service")
    try:
        reference_batches = resolve_champion_reference_batches(client, registered_model_name)
        print(f"Reference batches (champion's training data): {reference_batches}")

        reference_df = load_cleaned_batch(spark, reference_batches, repo_name)
        current_df = load_cleaned_batch(spark, [CURRENT_DATA_BATCH], repo_name)
    finally:
        spark.stop()

    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    result = report.as_dict()

    drift_result = result["metrics"][0]["result"]
    dataset_drift = drift_result["dataset_drift"]
    drift_share = drift_result["drift_share"]

    DATASET_DRIFT_DETECTED.set(1 if dataset_drift else 0)
    DRIFT_SHARE.set(drift_share)

    for col, col_result in drift_result.get("drift_by_columns", {}).items():
        COLUMN_DRIFT_SCORE.labels(column=col).set(col_result.get("drift_score", 0.0))

    report_path = "/tmp/drift_report.html"
    report.save_html(report_path)

    with mlflow.start_run(run_name="drift_check", experiment_id = 1):
        mlflow.log_param("reference_batches", " + ".join(reference_batches))
        mlflow.log_param("current_batch", CURRENT_DATA_BATCH)
        mlflow.log_metric("dataset_drift_detected", int(dataset_drift))
        mlflow.log_metric("drift_share", drift_share)
        mlflow.log_artifact(report_path)

    print(f"Drift check complete: dataset_drift={dataset_drift}, drift_share={drift_share:.3f}")
    LAST_CHECK_SUCCESS.set(1)

def trigger_retraining(pipeline_host, pipeline_name):
    import kfp
 
    client = kfp.Client(host=pipeline_host)
    pipeline_id = client.get_pipeline_id(name=pipeline_name)
    run = client.create_run_from_pipeline_id(
        pipeline_id=pipeline_id,
        run_name=f"drift-triggered-{int(time.time())}",
    )
    print(f"Drift detected -- triggered retraining run: {run.run_id}")

if __name__ == "__main__":
    start_http_server(8090)
    pipeline_host = os.environ.get("KFP_HOST", "http://ml-pipeline.kubeflow.svc.cluster.local:8888")
    pipeline_name = os.environ.get("TRAINING_PIPELINE_NAME", "training-pipeline")
 
    was_drifting = False  # edge-triggered: only fire once per drift episode, not every check while it persists
    while True:
        try:
            dataset_drift = run_drift_check()
            if dataset_drift and not was_drifting:
                trigger_retraining(pipeline_host, pipeline_name)
            was_drifting = dataset_drift
        except Exception as e:
            print(f"Drift check failed: {e}")
            LAST_CHECK_SUCCESS.set(0)
        time.sleep(DRIFT_CHECK_INTERVAL_SECONDS)