from typing import NamedTuple
from kfp.dsl import component


@component(
    base_image="tchaikovsky29/fraud-detection-base-env:latest"
)
def data_validation_component(commit_id: str, data_mode: str) -> NamedTuple(
    "ValidationOutput", [("validation_status", bool), 
                         ("message", str)]
):
    """
    Reads the schema of the raw batch(es) about to enter the cleaning stage
    (selected per DataValidationConfig.data_mode, same convention as the
    cleaning component) directly via Spark -- no separate metadata file --
    and checks column count, required columns, and target column presence
    against the schema YAML. Gates downstream components on the returned
    validation_status.
    """
    import sys
    import os
    from collections import namedtuple

    from src.exception import MyException
    from src.logger import get_logger
    from src.utils.main_utils import get_spark_session, read_yaml_file, get_shared_pipeline_run_id
    from src.entity.config_entity import DataValidationConfig, training_pipeline_config
    from src.configuration.lakefs_connection import fetch_batch, reset_branch

    try:
        ValidationOutput = namedtuple("ValidationOutput", ["validation_status", "message"])
        
        spark = get_spark_session("data-validation")
        os.environ["KFP_RUN_ID"] = get_shared_pipeline_run_id()
        logging = get_logger()
        reset_branch()
        config = DataValidationConfig()
        schema_config = read_yaml_file(file_path=training_pipeline_config.schema_path)

        batches = fetch_batch()
        if data_mode == "latest_only":
            paths = [f"s3a://{config.raw_data_path}/{batches[-1]}"]
        else:
            paths = [f"s3a://{config.raw_data_path}/{b}" for b in batches]

        logging.info(f"Validating schema for {len(paths)} batch(es) in mode {data_mode}: {paths}")

        # Schema-only read: Spark infers/reads the Parquet schema without
        # materializing row data, so this stays cheap even on the full
        # cumulative batch list.
        df_schema = spark.read.parquet(*paths).schema
        column_names = set(df_schema.names)
        column_count = len(df_schema.names)

        error_message = ""

        expected_count = len(schema_config["columns"])
        if column_count != expected_count:
            error_message += (
                f"Expected {expected_count} columns but found {column_count} columns.\n"
            )
        else:
            logging.info(f"Column count validation passed: {column_count} columns found.")

        missing_columns = [c for c in schema_config["columns"] if c not in column_names]
        if missing_columns:
            error_message += f"Missing columns: {', '.join(missing_columns)}.\n"
        else:
            logging.info("All required columns are present in the raw batch(es).")

        target_col = schema_config["target_column"]
        if target_col not in column_names:
            error_message += f"Target column '{target_col}' is missing in the raw batch(es).\n"
        else:
            logging.info(f"Target column '{target_col}' is present in the raw batch(es).")

        validation_status = len(error_message) == 0
        if validation_status:
            logging.info("Data validation passed.")
        else:
            logging.warning(f"Data validation failed: {error_message.strip()}")

        return ValidationOutput(validation_status, error_message.strip())
    except Exception as e:
        logging.error(f"Data validation error: {e}")
        raise MyException(e, sys)
    finally:
        spark.stop()