from typing import NamedTuple
from kfp.dsl import component

@component(
    base_image = "tchaikovsky29/fraud-detection-base-env:latest"
)
def data_cleaning_component(commit_id: str, data_mode: str, validation_status:bool, message:str) -> NamedTuple(
    "CleaningOutput", [("cleaned_data_path", str),
                       ("checksum", str),
                       ("data_used", str)]
):
    from src.entity.config_entity import DataCleaningConfig, training_pipeline_config
    from pyspark.sql import functions as F
    from pyspark.sql.types import IntegerType, FloatType
    from src.configuration.lakefs_connection import fetch_batch
    from src.utils.main_utils import get_spark_session, read_yaml_file, get_shared_pipeline_run_id
    from src.configuration.lakefs_connection import get_object_checksum
    from src.logger import get_logger
    from src.exception import MyException
    import sys
    import os

    try:
        os.environ["KFP_RUN_ID"] = get_shared_pipeline_run_id()
        logging = get_logger()
        batches = fetch_batch()
        config = DataCleaningConfig()
        schema_config = read_yaml_file(file_path=training_pipeline_config.schema_path)

        if not validation_status:
            logging.error(f"Data validation failed: {message}")
            raise MyException(f"Data validation failed: {message}", sys)
        
        spark = get_spark_session("data-cleaning")
        if data_mode == "latest_only":
            df = spark.read.parquet(f"s3a://{config.raw_data_path}/{batches[-1]}")
            data_used = batches[-1]
        else:
            df = spark.read.parquet(*[f"s3a://{config.raw_data_path}/{b}" for b in batches])
            data_used = " + ".join(batches)
        logging.info(f"Read {df.count()} rows in data mode {data_mode}")

        df = df.withColumn("Transaction Date", F.to_timestamp("Transaction Date"))
        df = df.withColumn("Transaction Day", F.dayofmonth("Transaction Date"))
        df = df.withColumn("Transaction DOW", (F.dayofweek("Transaction Date") + 5) % 7)
        df = df.withColumn("Transaction Month", F.month("Transaction Date"))
        logging.info("Extracted day, day of week, and month from Transaction Date.")

        mean_row = df.select(F.round(F.mean("Customer Age"), 0).alias("mean_age")).collect()[0]
        mean_age = float(mean_row["mean_age"])
        
        df = df.withColumn(
            "Customer Age",
            F.when(F.col("Customer Age") <= -9, F.abs(F.col("Customer Age"))).otherwise(
                F.col("Customer Age")
            ),
        )
        df = df.withColumn(
            "Customer Age",
            F.when(F.col("Customer Age") < 9, F.lit(mean_age)).otherwise(F.col("Customer Age")),
        )
        logging.info(f"Calculated mean Customer Age: {mean_age}. Replaced negative and missing values with this mean.")

        df = df.withColumn(
            "Is Address Match",
            (F.col("Shipping Address") == F.col("Billing Address")).cast("int"),
        )
        logging.info("Created Is Address Match column based on Shipping and Billing Address.")

        df = df.drop(*[c for c in schema_config["drop_columns"] if c in df.columns])
        logging.info(f"Dropped columns: {schema_config['drop_columns']}")

        for field in df.schema.fields:
            type_name = field.dataType.typeName()
            if type_name in ("long", "short", "byte"):
                df = df.withColumn(field.name, F.col(field.name).cast(IntegerType()))
            elif type_name == "double":
                df = df.withColumn(field.name, F.col(field.name).cast(FloatType()))
        logging.info("Casted numeric columns to appropriate types (int or float).")

        df.coalesce(1).write.mode("overwrite").parquet(f"s3a://{config.cleaned_data_path}")
        logging.info(f"Saved cleaned data to s3a://{config.cleaned_data_path} in parquet format.")
        checksum = get_object_checksum(config.cleaned_data_path)
        CleaningOutput = NamedTuple("CleaningOutput", [("cleaned_data_path", str), ("checksum", str), ("data_used", str)])
        return CleaningOutput(cleaned_data_path=config.cleaned_data_path, checksum=checksum, data_used=data_used)
    except Exception as e:
        raise MyException(e, sys)
    finally:
        spark.stop()