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
  