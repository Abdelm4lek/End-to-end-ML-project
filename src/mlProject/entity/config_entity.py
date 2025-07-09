from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    unzip_data_path: Path
    mode: str  # local or production
    # Local mode fields
    source_URL: str = None
    local_data_file: Path = None
    unzip_dir: Path = None
    # Production mode fields
    training_window_days: int = None
    min_data_points: int = None



@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict



@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path




@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    objective: str
    metric: str           
    boosting_type: str
    num_leaves: int
    learning_rate: float
    feature_fraction: float
    n_estimators: int
    target_column: str
    mlflow_uri: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str


@dataclass(frozen=True)
class PredictionConfig:
    root_dir: Path
    model_path: Path
    predictions_path: Path
    prediction_window_hours: int = 24
    min_data_points: int = 24  
    prediction_interval_minutes: int = 60