import os
from dataclasses import dataclass
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
    model_version: str = "1.0"

    def __post_init__(self):
        """Validate required configuration values."""
        if not self.api_key:
            raise ValueError("HOPSWORKS_API_KEY environment variable is not set")
        if not self.project_name:
            raise ValueError("HOPSWORKS_PROJECT_NAME environment variable is not set")



@dataclass
class MLflowBridgeConfig:
    """Configuration class for MLflow-Hopsworks bridge."""
    mlflow_tracking_uri: str
    model_name: str
    performance_threshold: Dict[str, float]
    hopsworks_config: HopsworksConfig