import os
import sys
import yaml
from src.exception import MyException
from pyspark.sql import SparkSession
from src.constants import *

def get_spark_session(app_name: str) -> SparkSession:
    lakefs_endpoint = LAKECTL_SERVER_ENDPOINT_URL
    lakefs_access_key = LAKECTL_CREDENTIALS_ACCESS_KEY_ID
    lakefs_secret_key = LAKECTL_CREDENTIALS_SECRET_ACCESS_KEY
 
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "6g")
        .config("spark.driver.maxResultSize", "2g")
        .config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.5.0",
        )
        .config("spark.hadoop.fs.s3a.endpoint", lakefs_endpoint)
        .config("spark.hadoop.fs.s3a.access.key", lakefs_access_key)
        .config("spark.hadoop.fs.s3a.secret.key", lakefs_secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .getOrCreate()
    )
    return spark

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise MyException(e, sys) from e

def get_shared_pipeline_run_id() -> str:
    """
    Extracts the shared Pipeline Run ID common to ALL pods in the active run.
    """
    hostname = os.environ.get('HOSTNAME', '')
    
    if hostname and '-' in hostname:
        parts = hostname.split('-')
        # Join 'training', 'pipeline', and the unique run hash
        if len(parts) >= 3:
            return f"{parts[0]}-{parts[1]}-{parts[2]}"

    return "unknown-run-id"

def calculate_cost(tn, fp, fn, tp):
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    total_financial_cost = (fn * COST_PER_MISSED_FRAUD) + (fp * COST_PER_BLOCKED_LEGIT_CUSTOMER)
    
    total_transactions = tn + fp + fn + tp
    average_cost_per_transaction = float(total_financial_cost / total_transactions) if total_transactions > 0 else 0.0
    return fn_rate, fp_rate, average_cost_per_transaction