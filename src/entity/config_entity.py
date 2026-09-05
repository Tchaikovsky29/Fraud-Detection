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
    learning_rate: float = 0.005187544455303851
    n_estimators: int = 1028
    max_depth: int = 2
    min_child_weight: int = 10
    gamma: float = 1.5116687959058668
    subsample: float = 0.8180714267188646
    colsample_bytree: float = 0.9926524228665097
    reg_alpha: float = 0.22366552184703914
    tree_method: str = "hist"

@dataclass
class ModelEvaluationConfig:
    max_acceptable_cost: float = 1.0
    
@dataclass
class ModelPusherConfig:
    registered_model_name: str = "fraud_detection_model"