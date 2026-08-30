from typing import NamedTuple
from kfp.dsl import component

@component(
    base_image="tchaikovsky29/fraud-detection-base-env:latest",
)
def model_training_component(
    train_path: str,
    mlflow_run_id: str,
) -> NamedTuple(
    "TrainingOutput",
    [
        ("model_uri", str)
    ],
):
    """
    Trains the notebook's winning model -- XGBoost -- using the exact
    hyperparameters found by the notebook's own Optuna search (70 trials,
    best test accuracy 0.9545). No evaluation, no promotion decision here
    -- that's the evaluation component's job.

    NOTE: the notebook used tree_method="gpu_hist" (GPU-accelerated).
    This environment has no GPU, so tree_method="hist" (CPU) is used
    instead -- same algorithm family, CPU execution.

    Logs the fitted model as an MLflow artifact under a nested child run
    (so evaluation can load it independently via model_run_id), nested
    under the parent run started in the transformation component.
    """
    import sys
    import os

    import numpy as np
    import mlflow
    import mlflow.xgboost
    from xgboost import XGBClassifier

    from src.utils.main_utils import get_spark_session, get_shared_pipeline_run_id, read_yaml_file
    from src.entity.config_entity import training_pipeline_config, ModelTrainingConfig
    from src.exception import MyException
    from src.logger import get_logger

    TrainingOutput = NamedTuple("TrainingOutput", [("model_uri", str)])

    spark = None
    try:
        os.environ["KFP_RUN_ID"] = get_shared_pipeline_run_id()
        logging = get_logger()
        target_col = read_yaml_file(training_pipeline_config.schema_path)["target_column"]
        config = ModelTrainingConfig()
        BEST_PARAMS = {
            "learning_rate": config.learning_rate,
            "n_estimators": config.n_estimators,
            "max_depth": config.max_depth,
            "min_child_weight": config.min_child_weight,
            "gamma": config.gamma,
            "subsample": config.subsample,
            "colsample_bytree": config.colsample_bytree,
            "reg_alpha": config.reg_alpha,
            "tree_method": config.tree_method,
        }

        spark = get_spark_session("model-training")

        train_df = spark.read.parquet(f"s3a://{train_path}")
        logging.info(f"Loaded train: {train_df.count()} rows.")

        # Spark's "features" column is a Vector -- convert to numpy for XGBoost's sklearn API
        pdf = train_df.select("features", target_col).toPandas()
        X_train = np.array(pdf["features"].apply(lambda v: v.toArray()).tolist())
        y_train = pdf[target_col].values

        import dagshub
        os.environ["DAGSHUB_USER_TOKEN"] = os.getenv("DAGSHUB_USER_TOKEN")
        dagshub.auth.add_app_token(os.getenv("DAGSHUB_USER_TOKEN"))
        dagshub.init(
            repo_owner=os.environ["DAGSHUB_REPO_OWNER"], repo_name=os.environ["DAGSHUB_REPO_NAME"], mlflow=True
        )

        with mlflow.start_run(run_id=mlflow_run_id):
            with mlflow.start_run(run_name="xgboost_final", nested=True) as run:
                model = XGBClassifier(**BEST_PARAMS)
                model.fit(X_train, y_train)

                mlflow.log_params(BEST_PARAMS)
                model_info = mlflow.xgboost.log_model(model, artifact_path="model")
                model_run_id = run.info.run_id
                model_uri = model_info.model_uri
                logging.info(f"Trained XGBoost, logged model under nested run {model_run_id}.")

        return TrainingOutput(model_uri=model_uri)

    except Exception as e:
        logging.error(f"Model training error: {e}")
        raise MyException(e, sys)
    finally:
        if spark is not None:
            spark.stop()