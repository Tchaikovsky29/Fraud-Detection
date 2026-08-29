from kfp.dsl import component

@component(
    base_image="tchaikovsky29/fraud-detection-base-env:latest",
)
def model_pusher_component(
    test_path: str,
    model_run_id: str,
    model_uri: str,
    mlflow_run_id: str,
):
    """
    Loads the model trained in model_training_component (via model_run_id)
    and transformation's test split, evaluates it, logs metrics to the
    parent MLflow run, and promotes to @champion in the registry if it
    beats the current champion's logged test_accuracy.

    NOTE: this evaluates against transformation's train/test split, NOT
    the true held-out set (the small Kaggle file) -- that's deferred to a
    future live-replay path: holdout transactions get fed through Kafka,
    scored by the deployed model, and evaluated from recorded predictions
    separately from this training pipeline.
    """
    import sys
    import os
    import numpy as np
    import mlflow
    import mlflow.xgboost
    from mlflow.tracking import MlflowClient
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
    from src.configuration.lakefs_connection import commit
    from src.utils.main_utils import get_spark_session, get_shared_pipeline_run_id, read_yaml_file
    from src.entity.config_entity import ModelPusherConfig, training_pipeline_config
    from src.exception import MyException
    from src.logger import get_logger

    try:
        os.environ["KFP_RUN_ID"] = get_shared_pipeline_run_id()
        logging = get_logger()
        config = ModelPusherConfig()
        target_col = read_yaml_file(training_pipeline_config.schema_path)["target_column"]

        spark = get_spark_session("model-pusher")

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
        cost = float(fn * config.cost_per_missed_fraud + fp * config.cost_per_blocked_legit_customer)
        logging.info(
            f"accuracy={accuracy:.4f} precision={precision:.4f} recall={recall:.4f} "
            f"FN={fn} FP={fp} cost={cost:.1f}"
        )

        with mlflow.start_run(run_id=mlflow_run_id):
            mlflow.log_metrics(
                {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "false_negatives": float(fn),
                    "false_positives": float(fp),
                    "cost": cost,
                }
            )

            # ---- Champion/challenger promotion -- LOWER cost wins ----
            client = MlflowClient()
            registered = mlflow.register_model(model_uri, config.registered_model_name)

            champion_promoted = False
            try:
                champion_version = client.get_model_version_by_alias(config.registered_model_name, "champion")
                champion_run = client.get_run(champion_version.run_id)
                parent_run_id = champion_run.data.tags.get("mlflow.parentRunId")
                champion_parent = client.get_run(parent_run_id)
                champion_cost = champion_parent.data.metrics.get("cost", float("inf"))
            except Exception:
                champion_cost = float("inf")  # no champion registered yet

            if cost < champion_cost:
                client.set_registered_model_alias(config.registered_model_name, "champion", registered.version)
                champion_promoted = True
                logging.info(
                    f"Promoted version {registered.version} to @champion "
                    f"(cost {cost:.1f} < previous {champion_cost:.1f})"
                )
            else:
                logging.info(
                    f"Version {registered.version} did NOT beat @champion "
                    f"(cost {cost:.1f} >= {champion_cost:.1f})"
                )
            mlflow.set_tag("champion_promoted", str(champion_promoted))
        commit_message = commit(message = f"data produced by kfp run: {os.environ.get('KFP_RUN_ID')}, mlflow_run_id: {mlflow_run_id}")
        if commit_message:
            logging.info(commit_message)
        else:
            logging.info("No new files, skipping commit")
    except Exception as e:
        logging.error(f"Model pusher error: {e}")
        raise MyException(e, sys)
    finally:
        if spark is not None:
            spark.stop()