"""
Simplified prediction service that reuses existing components.
"""

import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime
import os

from ..components.prediction import Predictor
from ..hopsworks.model_registry import HopsworksModelRegistry
from ..hopsworks.config import HopsworksConfig
from ..hopsworks.feature_store import HopsworksFeatureStore

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        """Initialize the prediction service using existing components."""
        self.config = HopsworksConfig()
        self.model_registry = HopsworksModelRegistry(self.config)
        self.feature_store = HopsworksFeatureStore(self.config)
        
        self.predictor = None
        self.model_loaded_at = None
        self.model_version = None
        
        # Try to load model on initialization
        asyncio.create_task(self._initialize_predictor())
    
    async def _initialize_predictor(self):
        """Initialize the predictor with model loading."""
        try:
            await self.reload_model()
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {str(e)}")
    
    async def reload_model(self) -> bool:
        """
        Reload the model using existing infrastructure.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            logger.info("Reloading model...")
            
            # Try to load from Hopsworks model registry first
            try:
                model = self.model_registry.load_model()
                if model is not None:
                    # Create predictor with the loaded model
                    # We need to temporarily save it to create the Predictor instance
                    temp_model_path = "temp_model.joblib"
                    import joblib
                    joblib.dump(model, temp_model_path)
                    
                    self.predictor = Predictor(model_path=temp_model_path)
                    # Clean up temp file
                    os.remove(temp_model_path)
                    
                    self.model_loaded_at = datetime.now()
                    self.model_version = self.config.model_version
                    logger.info(f"Successfully loaded model version {self.model_version} from Hopsworks")
                    return True
            except Exception as e:
                logger.warning(f"Failed to load from Hopsworks: {str(e)}")
            
            # Fallback to local model
            try:
                local_model_path = "artifacts/model_trainer/model.joblib"
                self.predictor = Predictor(model_path=local_model_path)
                self.model_loaded_at = datetime.now()
                self.model_version = "local_fallback"
                logger.info("Successfully loaded fallback model from local artifacts")
                return True
            except Exception as e:
                logger.error(f"Failed to load fallback model: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Error reloading model: {str(e)}")
            return False
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready for predictions."""
        return self.predictor is not None
    
    async def check_hopsworks_connection(self) -> bool:
        """
        Check if Hopsworks is accessible.
        
        Returns:
            bool: True if Hopsworks is accessible, False otherwise
        """
        try:
            # Try to get feature group to test connection
            from ..hopsworks.feature_schema import get_feature_group_schema
            schema = get_feature_group_schema()
            feature_group = self.feature_store.get_feature_group(
                name=schema["name"],
                version=schema["version"]
            )
            return feature_group is not None
        except Exception as e:
            logger.warning(f"Hopsworks connection check failed: {str(e)}")
            return False
    
    async def predict_all_stations(self) -> Dict[str, float]:
        """
        Get predictions for all stations using the existing Predictor class.
        
        Returns:
            Dict[str, float]: Dictionary mapping station names to predictions
        """
        try:
            if not self.is_model_loaded():
                logger.error("Model not loaded")
                return {}
            
            logger.info("Using existing Predictor.predict_all_stations() method")
            
            # Use the existing predict_all_stations method
            predictions = self.predictor.predict_all_stations()
            
            logger.info(f"Generated predictions for {len(predictions)} stations")
            return predictions
            
        except Exception as e:
            logger.error(f"Error in predict_all_stations: {str(e)}")
            return {}
    
    def get_model_info(self) -> Dict:
        """
        Get information about the currently loaded model.
        
        Returns:
            Dict: Model information
        """
        return {
            "model_loaded": self.is_model_loaded(),
            "model_version": self.model_version,
            "loaded_at": self.model_loaded_at.isoformat() if self.model_loaded_at else None,
            "model_type": type(self.predictor.model).__name__ if self.predictor and self.predictor.model else None,
            "predictor_class": "Predictor" if self.predictor else None
        } 