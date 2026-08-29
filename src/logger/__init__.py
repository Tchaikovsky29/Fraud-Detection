import logging
import os
import boto3
from logging.handlers import RotatingFileHandler
from datetime import datetime
from src.constants import BUCKET_NAME, MINIO_ENDPOINT_URL, MINIO_ACCESS_KEY_ID, MINIO_SECRET_ACCESS_KEY
from io import StringIO

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
LOG_DIR = "/tmp/logs"
MAX_LOG_SIZE = 5 * 1024 * 1024
BACKUP_COUNT = 2
FORMATTER = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

_logger_configured = False

# ─────────────────────────────────────────────
# S3 Handler
# ─────────────────────────────────────────────
class S3LogHandler(logging.Handler):
    """
    Buffers log records in memory and flushes them to S3 on close.
    The log file is uploaded to s3://<bucket>/logs/<log_file> when
    the handler is flushed — i.e. at the end of the component run.
    """

    def __init__(self, log_key: str):
        super().__init__()
        self.buffer = StringIO()
        self.s3_client = boto3.client(
            "s3",
            region_name="us-east-1",
            endpoint_url=MINIO_ENDPOINT_URL,
            aws_access_key_id=MINIO_ACCESS_KEY_ID,
            aws_secret_access_key=MINIO_SECRET_ACCESS_KEY,
        )
        self.bucket = BUCKET_NAME
        self.s3_key = log_key

    def emit(self, record):
        msg = self.format(record)
        self.buffer.write(msg + "\n")

    def flush_to_s3(self):
        if not self.bucket:
            return
        try:
            # Try to fetch existing log content first
            try:
                existing = self.s3_client.get_object(
                    Bucket=self.bucket,
                    Key=self.s3_key
                )
                existing_content = existing["Body"].read().decode("utf-8")
            except self.s3_client.exceptions.NoSuchKey:
                existing_content = ""

            # Append new content
            combined = existing_content + self.buffer.getvalue()
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=self.s3_key,
                Body=combined.encode("utf-8"),
            )
        except Exception as e:
            print(f"[Logger] Failed to upload logs to S3: {e}")

    def close(self):
        self.flush_to_s3()
        self.buffer.close()
        super().close()

def get_logger():
    global _logger_configured
    if _logger_configured:
        return logging.getLogger()

    run_id = os.getenv("KFP_RUN_ID", datetime.now().strftime('%m_%d_%Y_%H_%M_%S'))
    log_file_path = os.path.join(LOG_DIR, f"{run_id}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    os.makedirs(LOG_DIR, exist_ok=True)
    file_handler = RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT)
    file_handler.setFormatter(FORMATTER)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(FORMATTER)
    console_handler.setLevel(logging.INFO)

    s3_handler = S3LogHandler(log_key=f"logs/{run_id}.log")
    s3_handler.setFormatter(FORMATTER)
    s3_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(s3_handler)

    for noisy in ("botocore", "boto3", "urllib3", "dagshub", "httpcore", "httpx", "mlflow", "skl2onnx", "git", "py4j"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _logger_configured = True
    return logger