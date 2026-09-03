import os
import json
import uuid
from dotenv import load_dotenv
import boto3
import pandas as pd
import requests
import lakefs
from kafka import KafkaConsumer

load_dotenv()
REPO_NAME = os.environ.get("LAKEFS_REPO_NAME", "fraud-detection")
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9094")
TOPIC = "transactions"
PREDICTOR_URL = os.environ.get(
    "PREDICTOR_URL", "http://localhost:8888/v1/models/fraud-detector:predict"
)
FLUSH_EVERY = int(os.environ.get("FLUSH_EVERY", "500"))


def flush_predictions(buffer):
    if not buffer:
        return
    df = pd.DataFrame(buffer)
    local_path = f"/tmp/predictions_{uuid.uuid4().hex[:8]}.parquet"
    df.to_parquet(local_path, index=False)

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["MINIO_ENDPOINT_URL"],
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["MINIO_SECRET_ACCESS_KEY"],
    )
    key = f"predictions/{uuid.uuid4()}.parquet"
    s3.upload_file(local_path, os.environ["BUCKET_NAME"], key)
    print(f"Flushed {len(buffer)} prediction records to s3://{os.environ['BUCKET_NAME']}/{key}")

def main():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
    )

    raw_rows = []
    prediction_buffer = []

    for message in consumer:
        record = message.value

        if record.get("event") == "END_OF_STREAM":
            print("Reached end of stream.")
            break

        raw_rows.append(record)

        response = requests.post(PREDICTOR_URL, json={"instances": [record]}, timeout=30)
        response.raise_for_status()
        prediction = response.json()["predictions"][0]

        prediction_buffer.append({
            "transaction_id": record.get("Transaction ID"),
            "transaction_date": record.get("Transaction Date"),
            "is_fraud_predicted": prediction["is_fraud"],
            "fraud_probability": prediction["fraud_probability"],
            "top_shap_factors": json.dumps(prediction["top_shap_factors"]),
            "actual_is_fraud": record.get("Is Fraudulent"),  # holdout has real labels -- enables true evaluation later
        })

        if len(prediction_buffer) >= FLUSH_EVERY:
            flush_predictions(prediction_buffer)
            prediction_buffer = []

    flush_predictions(prediction_buffer)  # flush any remainder under the batch size

    print(f"Building batch-4.parquet from {len(raw_rows)} consumed transactions...")
    raw_df = pd.DataFrame(raw_rows)
    # Same timestamp-precision fix as the original ingestion script --
    # Spark can't read nanosecond-precision Parquet timestamps.
    raw_df["Transaction Date"] = pd.to_datetime(raw_df["Transaction Date"]).astype("datetime64[us]")

    repo = lakefs.Repository(REPO_NAME)
    branch = repo.branch("main")
    branch.object("raw/batch-4.parquet").upload(raw_df.to_parquet(index=False), pre_sign=False)
    try:
        branch.commit(message="raw: batch 4 (Kafka-replayed holdout set, post live-evaluation)")
        print("Committed batch-4.parquet to lakeFS main branch.")
    except Exception as e:
        print("commit failed. No changes to commit")

if __name__ == "__main__":
    main()