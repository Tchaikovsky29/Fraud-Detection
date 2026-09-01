from kfp import kubernetes
import kfp
from kfp.dsl import pipeline
from kfp.kubernetes import use_secret_as_env
from src.components.commit_retrieval import resolve_commit_component
from src.components.data_validation import data_validation_component
from src.components.data_cleaning import data_cleaning_component
from src.components.data_transformation import data_transformation_component
from src.components.model_trainer import model_training_component
from src.components.model_evaluation import model_evaluation_component
from src.components.model_pusher import model_pusher_component
import datetime

SECRET_KEYS = {
    "LAKECTL_SERVER_ENDPOINT_URL": "LAKECTL_SERVER_ENDPOINT_URL",
    "LAKECTL_CREDENTIALS_ACCESS_KEY_ID": "LAKECTL_CREDENTIALS_ACCESS_KEY_ID",
    "LAKECTL_CREDENTIALS_SECRET_ACCESS_KEY": "LAKECTL_CREDENTIALS_SECRET_ACCESS_KEY",
    "LAKEFS_REPO_NAME": "LAKEFS_REPO_NAME",
    "MAIN_BRANCH_NAME": "MAIN_BRANCH_NAME",
    "FEATURES_BRANCH_NAME": "FEATURES_BRANCH_NAME",
    "DAGSHUB_REPO_OWNER": "DAGSHUB_REPO_OWNER",
    "DAGSHUB_REPO_NAME": "DAGSHUB_REPO_NAME",
    "DAGSHUB_USER_TOKEN": "DAGSHUB_USER_TOKEN",
    "MINIO_ENDPOINT_URL": "MINIO_ENDPOINT_URL",
    "MINIO_ACCESS_KEY_ID": "MINIO_ACCESS_KEY_ID",
    "MINIO_SECRET_ACCESS_KEY": "MINIO_SECRET_ACCESS_KEY",
    "BUCKET_NAME": "BUCKET_NAME",
}

def apply_secrets(task):
    return use_secret_as_env(task, secret_name="pipeline-secrets", secret_key_to_env=SECRET_KEYS)

def configure_task(task):
    """Applies common configuration to every task."""
    apply_secrets(task)
    kubernetes.set_image_pull_policy(task, "Always")
    return task

@pipeline(
    name="training-pipeline",
    description="Runs the training pipeline for fraud detection"
)
def training_pipeline():
    get_commit = resolve_commit_component()
    get_commit.set_caching_options(False)
    configure_task(get_commit)

    validate = data_validation_component(
        commit_id = get_commit.outputs["commit_id"],
        data_mode = get_commit.outputs["data_mode"],
    )
    configure_task(validate)

    clean = data_cleaning_component(
        commit_id = get_commit.outputs["commit_id"],
        data_mode = get_commit.outputs["data_mode"],
        validation_status = validate.outputs["validation_status"],
        message = validate.outputs["message"],
    )
    configure_task(clean)

    transform = data_transformation_component(
        cleaned_data_path = clean.outputs["cleaned_data_path"],
        cleaned_data_checksum = clean.outputs["checksum"],
        data_mode = get_commit.outputs["data_mode"],
        data_used=clean.outputs["data_used"]
    )
    configure_task(transform)

    train = model_training_component(
        train_path = transform.outputs["train_path"],
        mlflow_run_id = transform.outputs["mlflow_run_id"]
    )
    configure_task(train)

    evaluate = model_evaluation_component(
        test_path = transform.outputs["test_path"],
        mlflow_run_id = transform.outputs["mlflow_run_id"],
        model_uri = train.outputs["model_uri"]
    )
    configure_task(evaluate)

    pusher = model_pusher_component(
        test_path = transform.outputs["test_path"],
        mlflow_run_id = transform.outputs["mlflow_run_id"],
        challenger_cost= evaluate.outputs["cost"],
        gate_passed= evaluate.outputs["gate_passed"],
        model_uri = train.outputs["model_uri"]
    )
    configure_task(pusher)

if __name__ == "__main__":
    try:
        client = kfp.Client(host="http://localhost:8888/")
        pipeline_id = client.get_pipeline_id(name="training-pipeline")
        if not pipeline_id:
            client.upload_pipeline_from_pipeline_func(
                pipeline_func=training_pipeline,
                pipeline_name="training-pipeline"
            )
        else:
            client.upload_pipeline_version_from_pipeline_func(
                pipeline_func=training_pipeline,
                pipeline_version_name=f"training-pipeline-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
                pipeline_name="training-pipeline"
            )
    except Exception as e:
        print(f"Error retrieving pipeline versions: {e}")