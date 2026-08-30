from kfp.dsl import component

@component(
    base_image="tchaikovsky29/fraud-detection-base-env:latest",
)
def model_pusher_component(
    test_path: str,
    model_uri: str,
    challenger_cost: float,
    gate_passed: bool,
    mlflow_run_id: str,
):
    import sys
    import os

    import numpy as np
    import mlflow
    import mlflow.xgboost
    from mlflow.tracking import MlflowClient
    from sklearn.metrics import confusion_matrix, recall_score
    from src.configuration.lakefs_connection import commit
    from src.utils.main_utils import get_spark_session, get_shared_pipeline_run_id, calculate_cost, read_yaml_file
    from src.entity.config_entity import ModelPusherConfig, training_pipeline_config
    from src.exception import MyException
    from src.logger import get_logger

    spark = None
    try:
        os.environ["KFP_RUN_ID"] = get_shared_pipeline_run_id()
        logging = get_logger()
        config = ModelPusherConfig()
        target_col = read_yaml_file(training_pipeline_config.schema_path)["target_column"]

        if not gate_passed:
            logging.info("Challenger did not pass the absolute quality gate -- skipping promotion entirely.")
            return

        spark = get_spark_session("model-pusher")

        import dagshub
        os.environ["DAGSHUB_USER_TOKEN"] = os.getenv("DAGSHUB_USER_TOKEN")
        dagshub.auth.add_app_token(os.getenv("DAGSHUB_USER_TOKEN"))
        dagshub.init(
            repo_owner=os.environ["DAGSHUB_REPO_OWNER"], repo_name=os.environ["DAGSHUB_REPO_NAME"], mlflow=True
        )

        client = MlflowClient()
        model_uri_for_registration = model_uri
        registered = mlflow.register_model(model_uri_for_registration, config.registered_model_name)

        champion_cost = float("inf")
        try:
            champion_version = client.get_model_version_by_alias(config.registered_model_name, "champion")
            champion_run = client.get_run(champion_version.run_id)
            champion_model_uri = f"models:/{champion_run.outputs.model_outputs[0].model_id}"
            champion_model = mlflow.xgboost.load_model(champion_model_uri)

            # Re-evaluate the champion fresh, on this run's test set
            test_df = spark.read.parquet(f"s3a://{test_path}")
            pdf = test_df.select("features", target_col).toPandas()
            X_test = np.array(pdf["features"].apply(lambda v: v.toArray()).tolist())
            y_test = pdf[target_col].values

            champion_preds = champion_model.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, champion_preds).ravel()
            fn_rate, fp_rate, champion_cost = calculate_cost(tn, fp, fn, tp)
            logging.info(f"Re-evaluated current champion (v{champion_version.version}) on this test set: cost={champion_cost:.4f}")
        except Exception as lookup_err:
            logging.info(f"No existing champion to compare against ({lookup_err}); treating as unbounded cost.")

        champion_promoted = False
        with mlflow.start_run(run_id=mlflow_run_id):
            if challenger_cost < champion_cost:
                client.set_registered_model_alias(config.registered_model_name, "champion", registered.version)
                champion_promoted = True
                logging.info(
                    f"Promoted version {registered.version} to @champion "
                    f"(challenger cost {challenger_cost:.4f} < champion cost {champion_cost:.4f})"
                )
            else:
                logging.info(
                    f"Version {registered.version} did NOT beat @champion "
                    f"(challenger cost {challenger_cost:.4f} >= champion cost {champion_cost:.4f})"
                )
            mlflow.set_tag("champion_promoted", str(champion_promoted))
        if champion_promoted:
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