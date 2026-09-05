import io
import os
from dotenv import load_dotenv
import boto3
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

load_dotenv()
REPO_NAME = os.environ.get("LAKEFS_REPO_NAME", "fraud-detection")
from src.utils.main_utils import calculate_cost

OUTPUT_LOCAL_PATH = "/tmp/joined_holdout_predictions.parquet"
OUTPUT_MINIO_KEY = "demo/joined_holdout_predictions.parquet"

def fetch_all_predictions():
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["MINIO_ENDPOINT_URL"],
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["MINIO_SECRET_ACCESS_KEY"],
    )
    bucket = os.environ["BUCKET_NAME"]

    paginator = s3.get_paginator("list_objects_v2")
    dfs = []
    for page in paginator.paginate(Bucket=bucket, Prefix="predictions/"):
        for obj in page.get("Contents", []):
            body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
            dfs.append(pd.read_parquet(io.BytesIO(body)))

    if not dfs:
        raise RuntimeError("No prediction files found under predictions/ -- did the consumer run?")

    return pd.concat(dfs, ignore_index=True)

def main():
    print("Fetching predictions from MinIO...")
    predictions_df = fetch_all_predictions()
    print(f"Loaded {len(predictions_df)} prediction records.")

    # --- True holdout metrics ---
    y_true = predictions_df["actual_is_fraud"].astype(int)
    y_pred = predictions_df["is_fraud_predicted"].astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fn_rate, fp_rate, cost = calculate_cost(tn, fp, fn, tp)

    print("\n=== TRUE HOLDOUT EVALUATION (live Kafka-replayed predictions) ===")
    print(f"accuracy:  {accuracy:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall:    {recall:.4f}")
    print(f"fn_rate:   {fn_rate:.4f}")
    print(f"fp_rate:   {fp_rate:.4f}")
    print(f"cost:      {cost:.4f}")

if __name__ == "__main__":
    main()