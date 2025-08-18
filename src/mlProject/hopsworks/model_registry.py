import os
from src.mlProject import logger
import hopsworks
import joblib
from typing import Optional, Any
from src.mlProject.hopsworks.config import HopsworksConfig

class HopsworksModelRegistry:
    def __init__(self, config: HopsworksConfig):
        """Initialize connection to Hopsworks model registry."""
        self.config = config
        self.project = hopsworks.login(
            api_key_value=self.config.api_key,
            project=self.config.project_name
        )
        self.mr = self.project.get_model_registry()
    
    def save_model(self, model: Any, metrics: dict, description: str = "Velib demand prediction model"):
        """Save a model to the model registry."""
        
        try:
            # Create model registry entry
            logger.info(f"Creating model registry entry for {self.config.model_name}")
            model_registry_entry = self.mr.python.create_model(
                name=self.config.model_name,
                metrics=metrics,
                description=description
            )
            
            # Save model to the model directory first
            model_path = f"artifacts/model_trainer/model.joblib"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            logger.info(f"Saving model to local path: {model_path}")
            joblib.dump(model, model_path)
            
            # Save model to Hopsworks registry
            logger.info("Uploading model to Hopsworks registry")
            model_registry_entry.save(model_path)
            logger.info("Model successfully saved to Hopsworks registry")
            
        except Exception as e:
            logger.error(f"Failed to save model to Hopsworks registry: {str(e)}")
            raise e

    
    def load_model(self) -> Optional[Any]:
        """Download and load the latest model from the Hopsworks registry."""
        try:
            # Hopsworks API expects an integer version; ensure we pass one
            try:
                version_int = int(self.config.model_version) if self.config.model_version else None
            except ValueError:
                logger.warning(f"Invalid model_version '{self.config.model_version}', will fetch latest version instead")
                version_int = None

            if version_int is not None:
                model_obj = self.mr.get_model(name=self.config.model_name, version=version_int)
            else:
                # No version provided or invalid – fetch latest
                model_obj = self.mr.get_model(name=self.config.model_name)

            # Download the model artifacts to a temporary local directory
            local_dir = model_obj.download()
            model_path = os.path.join(local_dir, "model.joblib")
            logger.info(f"Loaded model artifact from: {model_path}")
            return joblib.load(model_path)
        except Exception as e:
            logger.error(f"Failed to load model from Hopsworks registry: {e}")
            return None
    
    def get_model_metrics(self) -> Optional[dict]:
        """Get metrics for the latest model version."""
        try:
            model = self.mr.get_model(
                name=self.config.model_name,
                version=self.config.model_version
            )
            return model.metrics
        except:
            return None
    
    def list_model_versions(self):
        """List all versions of the model."""
        return self.mr.get_model_versions(self.config.model_name) 