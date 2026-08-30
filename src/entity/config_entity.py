import os
from src.constants import *
from dataclasses import dataclass

@dataclass
class TrainingPipelineConfig:
    main_data_path: str = os.path.join(LAKEFS_REPO_NAME, MAIN_BRANCH_NAME)
    features_data_path: str = os.path.join(LAKEFS_REPO_NAME, FEATURES_BRANCH_NAME)
    schema_path: str = 'config/schema.yaml'
training_pipeline_config = TrainingPipelineConfig()

@dataclass
class DataValidationConfig:
    raw_data_path: str = os.path.join(training_pipeline_config.main_data_path)

@dataclass
class DataCleaningConfig:
    """
    Configuration for the data cleaning component.
    """
    raw_data_path: str = os.path.join(training_pipeline_config.main_data_path)
    cleaned_data_path: str = os.path.join(training_pipeline_config.features_data_path, CLEANED_DATA_DIR, "cleaned_data.parquet")

@dataclass
class DataTransformationConfig:
    """
    Configuration for the data transformation component.
    """
    raw_data_path: str = os.path.join(training_pipeline_config.main_data_path, RAW_DATA_DIR)
    train_path: str = os.path.join(training_pipeline_config.features_data_path, TRANSFORMED_DATA_DIR, "training_data.parquet")
    test_path: str = os.path.join(training_pipeline_config.features_data_path, TRANSFORMED_DATA_DIR, "test_data.parquet")
    test_size: float = 0.2 

@dataclass
class ModelTrainingConfig:
    learning_rate: float = 0.2381413287859572
    n_estimators: int = 439
    max_depth: int = 6
    min_child_weight: int = 3
    gamma: float = 0.3308509875421135
    subsample: float = 0.9642190731549022
    colsample_bytree: float = 0.40885708517990055
    reg_alpha: float = 0.35435676335715693
    tree_method: str = "hist"

@dataclass
class ModelEvaluationConfig:
    max_acceptable_cost: float = 5

@dataclass
class ModelPusherConfig:
    registered_model_name: str = "fraud_detection_model"