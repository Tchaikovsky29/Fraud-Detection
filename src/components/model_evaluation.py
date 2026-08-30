from typing import NamedTuple
from kfp.dsl import component

@component(
    base_image="tchaikovsky29/fraud-detection-base-env:latest",
)
def model_evaluation_component(
    test_path: str,
    model_uri: str,
    mlflow_run_id: str,
) -> NamedTuple(
    "EvaluationOutput",
    [
        ("cost", float),
        ("gate_passed", bool),
    ],
):
    import sys
    import os

    import numpy as np
    import mlflow
    import mlflow.xgboost
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

    from src.utils.main_utils import get_spark_session, get_shared_pipeline_run_id, read_yaml_file, calculate_cost
    from src.entity.config_entity import ModelEvaluationConfig, training_pipeline_config
    from src.exception import MyException
    from src.logger import get_logger

    EvaluationOutput = NamedTuple(
        "EvaluationOutput",
        [("cost", float), ("gate_passed", bool)],
    )

    spark = None
    try:
        os.environ["KFP_RUN_ID"] = get_shared_pipeline_run_id()
        logging = get_logger()
        config = ModelEvaluationConfig()
        target_col = read_yaml_file(training_pipeline_config.schema_path)["target_column"]

        spark = get_spark_session("model-evaluation")

        test_df = spark.read.parquet(f"s3a://{test_path}")
        logging.info(f"Loaded test: {test_df.count()} rows.")

        pdf = test_df.select("features", target_col).toPandas()
        X_test = np.array(pdf["features"].apply(lambda v: v.toArray()).tolist())
        y_test = pdf[target_col].values

        import dagshub
        os.environ["DAGSHUB_USER_TOKEN"] = os.getenv("DAGSHUB_USER_TOKEN")
        dagshub.auth.add_app_token(os.getenv("DAGSHUB_USER_TOKEN"))
        dagshub.init(
            repo_owner=os.environ["DAGSHUB_REPO_OWNER"], repo_name=os.environ["DAGSHUB_REPO_NAME"], mlflow=True
        )

        model = mlflow.xgboost.load_model(model_uri)
        y_pred = model.predict(X_test)

        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fn_rate, fp_rate, cost = calculate_cost(tn, fp, fn, tp)

        gate_passed = cost <= config.max_acceptable_cost
        logging.info(
            f"accuracy={accuracy:.4f} precision={precision:.4f} recall={recall:.4f} "
            f"FN_rate={fn_rate:.4f} FP_rate={fp_rate:.4f} cost={cost:.4f} gate_passed={gate_passed}"
        )

        with mlflow.start_run(run_id=mlflow_run_id):
            mlflow.log_metrics(
                {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "fn_rate": fn_rate,
                    "fp_rate": fp_rate,
                    "cost": cost,
                }
            )
            mlflow.set_tag("gate_passed", str(gate_passed))

        return EvaluationOutput(
            cost=cost, gate_passed=gate_passed
        )

    except Exception as e:
        logging.error(f"Model evaluation error: {e}")
        raise MyException(e, sys)
    finally:
        if spark is not None:
            spark.stop()