from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    unzip_data_path: Path



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


@dataclass(frozen=True)
class PredictionConfig:
    model_path: Path
    prediction_window_hours: int = 24
    min_data_points: int = 24  
    prediction_interval_minutes: int = 60