import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
from typing import Dict

# Load environment variables
load_dotenv('DB_credentials.env')


@dataclass
class HopsworksConfig:
    """Configuration class for Hopsworks integration."""
    api_key: str = os.getenv("HOPSWORKS_API_KEY")
    project_name: str = os.getenv("HOPSWORKS_PROJECT_NAME")
    host: str = os.getenv("HOPSWORKS_HOST")
    port: int = os.getenv("HOPSWORKS_PORT")
    feature_group_name: str = "Velib_data_features"
    feature_view_name: str = "Velib_data_feature_view"
    model_name: str = "Velib_demand_model"
    # Hopsworks expects an integer version identifier
    model_version: int = int(os.getenv("HOPSWORKS_MODEL_VERSION", "1"))

    def __post_init__(self):
        """Validate required configuration values."""
        if not self.api_key:
            raise ValueError("HOPSWORKS_API_KEY environment variable is not set")
        if not self.project_name:
            raise ValueError("HOPSWORKS_PROJECT_NAME environment variable is not set")



@dataclass
class MLflowBridgeConfig:
    """Configuration class for MLflow-Hopsworks bridge."""
    mlflow_tracking_uri: str = "https://dagshub.com/Abdelm4lek/End-to-end-ML-project.mlflow"
    model_name: str = "Velib_demand_model"
    performance_threshold: Dict[str, float] = field(default_factory=lambda: {
        "rmse": 2.0,
        "mae": 1.5,
        "r2": 0.90
    })
    hopsworks_config: HopsworksConfig = field(default_factory=HopsworksConfig)