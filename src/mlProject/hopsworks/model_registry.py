import hopsworks
import joblib
from typing import Optional, Any
from .config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    MODEL_NAME,
    MODEL_VERSION
)

class HopsworksModelRegistry:
    def __init__(self):
        """Initialize connection to Hopsworks model registry."""
        self.project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT_NAME
        )
        self.mr = self.project.get_model_registry()
    
    def save_model(self, model: Any, metrics: dict, description: str = "Velib demand prediction model"):
        """Save a model to the model registry."""
        model_dir = self.mr.create_model(
            name=MODEL_NAME,
            metrics=metrics,
            description=description
        )
        
    # Save model to the model directory
        model_path = f"{model_dir}/model.joblib"
        joblib.dump(model, model_path)
        
        # Create a new model version
        self.mr.create_model_version(
            name=MODEL_NAME,
            version=MODEL_VERSION,
            metrics=metrics,
            description=description
        )
    
    def load_model(self) -> Optional[Any]:
        """Load the latest model from the model registry."""
        try:
            model = self.mr.get_model(
                name=MODEL_NAME,
                version=MODEL_VERSION
            )
            return joblib.load(f"{model.model_dir}/model.joblib")
        except:
            return None
    
    def get_model_metrics(self) -> Optional[dict]:
        """Get metrics for the latest model version."""
        try:
            model = self.mr.get_model(
                name=MODEL_NAME,
                version=MODEL_VERSION
            )
            return model.metrics
        except:
            return None
    
    def list_model_versions(self):
        """List all versions of the model."""
        return self.mr.get_model_versions(MODEL_NAME) 