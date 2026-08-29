from typing import NamedTuple
from kfp.dsl import component

@component(
    base_image="tchaikovsky29/fraud-detection-base-env:latest",
)
def data_transformation_component(
    cleaned_data_path: str,
    cleaned_data_checksum: str,   # unused in logic; forces correct KFP cache-busting
    data_mode: str,
    data_used: str
) -> NamedTuple(
    "TransformationOutput",
    [
        ("train_path", str),
        ("test_path", str),
        ("mlflow_run_id", str),
    ],
):
    """
    Loads cleaned Parquet from lakeFS and:
      - Fits a PySpark ML Pipeline (StringIndexer + OneHotEncoder on
        categorical columns, StandardScaler on numeric columns) -- mirrors
        the notebook's sklearn ColumnTransformer
      - Splits into train/test sets
      - Writes train/test Parquet + the fitted PipelineModel to the
        `features` branch (staged, not committed -- committed once at the
        end of the full pipeline run)
      - Logs transformation params to MLflow
    """
    import sys
    import os
    from collections import namedtuple

    import mlflow
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler

    from src.utils.main_utils import get_spark_session, read_yaml_file, get_shared_pipeline_run_id
    from src.configuration.lakefs_connection import get_object_checksum
    from src.entity.config_entity import DataTransformationConfig, training_pipeline_config
    from src.exception import MyException
    from src.logger import get_logger

    TransformationOutput = namedtuple(
        "TransformationOutput",
        ["train_path", "test_path", "mlflow_run_id"],
    )

    spark = get_spark_session("data-transformation")

    try:
        os.environ["KFP_RUN_ID"] = get_shared_pipeline_run_id()
        logging = get_logger()
        config = DataTransformationConfig()
        schema = read_yaml_file(training_pipeline_config.schema_path)

        categorical_cols = schema["categorical_columns"]
        numeric_cols = schema["numeric_columns"]          # scaled via StandardScaler
        passthrough_cols = schema["passthrough_columns"]  # already-binary flags, not scaled
        target_col = schema["target_column"]

        logging.info(f"Loading cleaned dataset from {cleaned_data_path}...")
        df = spark.read.parquet(f"s3a://{cleaned_data_path}")
        logging.info(f"Loaded cleaned dataset: {df.count()} rows")

        # --- Build + fit the preprocessing pipeline ---
        indexed_cols = [f"{c}_idx" for c in categorical_cols]
        encoded_cols = [f"{c}_ohe" for c in categorical_cols]

        stages = []
        for cat_col, idx_col in zip(categorical_cols, indexed_cols):
            stages.append(StringIndexer(inputCol=cat_col, outputCol=idx_col, handleInvalid="keep"))
        stages.append(OneHotEncoder(inputCols=indexed_cols, outputCols=encoded_cols))
        stages.append(VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features"))
        stages.append(
            StandardScaler(
                inputCol="numeric_features",
                outputCol="scaled_numeric_features",
                withMean=True,
                withStd=True,
            )
        )
        stages.append(
            VectorAssembler(
                inputCols=encoded_cols + ["scaled_numeric_features"] + passthrough_cols,
                outputCol="features",
            )
        )

        pipeline = Pipeline(stages=stages)
        pipeline_model = pipeline.fit(df)
        logging.info("Fitted preprocessing pipeline (StringIndexer + OneHotEncoder + StandardScaler).")

        transformed_df = pipeline_model.transform(df).select("features", target_col)

        # --- Train/test split ---
        train_df, test_df = transformed_df.randomSplit(
            [1 - config.test_size, config.test_size], seed=42
        )
        train_count, test_count = train_df.count(), test_df.count()
        logging.info(f"Train/test split: {train_count} train rows, {test_count} test rows")

        train_df.coalesce(1).write.mode("overwrite").parquet(f"s3a://{config.train_path}")
        test_df.coalesce(1).write.mode("overwrite").parquet(f"s3a://{config.test_path}")
        logging.info(f"Wrote train -> {config.train_path}, test -> {config.test_path}")

        import dagshub
        os.environ["DAGSHUB_USER_TOKEN"] = os.getenv("DAGSHUB_USER_TOKEN")
        dagshub.auth.add_app_token(os.getenv("DAGSHUB_USER_TOKEN"))
        dagshub.init(repo_owner=os.environ["DAGSHUB_REPO_OWNER"], repo_name=os.environ["DAGSHUB_REPO_NAME"], mlflow=True)
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logging.info(f"Started MLflow run: {run_id}")
            mlflow.log_params(
                {
                    "encoding": "StringIndexer+OneHotEncoder",
                    "scaling": "StandardScaler",
                    "test_size": config.test_size,
                    "random_state": 42,
                    "train_rows": train_count,
                    "test_rows": test_count,
                    "data_mode": data_mode,
                    "cleaned_data_checksum": cleaned_data_checksum,
                    "kfp_run_id": os.environ.get("KFP_RUN_ID", "N/A"),
                    "data_used": data_used
                }
            )
            mlflow.spark.log_model(pipeline_model, artifact_path="pipeline_model")
            logging.info("Logged preprocessing pipeline model to MLflow.")

        spark.stop()
        return TransformationOutput(
            train_path=config.train_path,
            test_path=config.test_path,
            mlflow_run_id=run_id,
        )
    except Exception as e:
        logging.error(f"Data transformation error: {e}")
        raise MyException(e, sys)