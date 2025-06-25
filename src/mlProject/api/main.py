"""
Simplified FastAPI application for hourly batch Velib predictions.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import logging
from datetime import datetime
import uvicorn
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from .prediction_service import PredictionService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Velib Hourly Prediction API",
    description="Hourly batch bike demand predictions for all Velib stations in Paris",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction service
prediction_service = PredictionService()

# Pydantic models
class BatchPredictionResponse(BaseModel):
    predictions: Dict[str, float]
    total_stations: int
    successful_predictions: int
    timestamp: datetime
    status: str = "success"
    model_info: Optional[Dict] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    model_loaded: bool
    hopsworks_connected: bool

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "title": "Velib bike hourly demand prediction API",
        "description": "Batch predictions for all Velib stations in Ile-de-France",
        "version": "1.0.0",
        "endpoints": {
            "predictions": "/predict/all",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        model_loaded = prediction_service.is_model_loaded()
        hopsworks_connected = await prediction_service.check_hopsworks_connection()
        
        return HealthResponse(
            status="healthy" if model_loaded else "degraded",
            timestamp=datetime.now(),
            model_loaded=model_loaded,
            hopsworks_connected=hopsworks_connected
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/predict/all", response_model=BatchPredictionResponse)
async def predict_all_stations():
    """
    Get hourly predictions for all Velib stations.
    
    This endpoint uses prediction pipeline to generate demand forecasts for the next hour for every station in the network.
    """
    try:
        logger.info("Starting batch prediction for all stations")
        
        # Use existing prediction service
        predictions = await prediction_service.predict_all_stations()
        
        if not predictions:
            raise HTTPException(
                status_code=503, 
                detail="Could not generate predictions - insufficient data or model not available"
            )
        
        # Get model information
        model_info = prediction_service.get_model_info()
        
        response = BatchPredictionResponse(
            predictions=predictions,
            total_stations=len(predictions),
            successful_predictions=len([p for p in predictions.values() if p is not None]),
            timestamp=datetime.now(),
            status="success",
            model_info=model_info
        )
        
        logger.info(f"Batch prediction completed: {response.successful_predictions}/{response.total_stations} stations")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/model/reload")
async def reload_model():
    """Reload the prediction model."""
    try:
        success = await prediction_service.reload_model()
        
        if success:
            return {
                "message": "Model reloaded successfully",
                "timestamp": datetime.now(),
                "model_info": prediction_service.get_model_info()
            }
        else:
            raise HTTPException(status_code=503, detail="Failed to reload model")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 