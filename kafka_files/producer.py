import os
import io
import json
import time

import pandas as pd
import lakefs
from kafka import KafkaProducer

REPO_NAME = os.environ.get("LAKEFS_REPO_NAME", "fraud-detection")
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9094")
TOPIC = "transactions"
DELAY_SECONDS = float(os.environ.get("PRODUCER_DELAY_SECONDS", "0.05"))

def main():
    repo = lakefs.Repository(REPO_NAME)
    obj = repo.branch("main").object("holdout/eval.parquet")
    with obj.reader(pre_sign=False) as f:  # verify .reader() matches your installed lakefs SDK version
        df = pd.read_parquet(io.BytesIO(f.read()))

    print(f"Loaded {len(df)} holdout rows. Streaming to Kafka topic '{TOPIC}'...")

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
    )

    count = 0
    for _, row in df.iterrows():
        if count >= 500:
            break
        count += 1
        producer.send(TOPIC, value=row.to_dict())
        time.sleep(DELAY_SECONDS)

    producer.send(TOPIC, value={"event": "END_OF_STREAM"})
    producer.flush()
    print(f"Finished streaming {count} rows.")

if __name__ == "__main__":
    main()