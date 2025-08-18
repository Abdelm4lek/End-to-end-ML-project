"""
Simplified FastAPI application for hourly batch Velib predictions.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List
import logging
from datetime import datetime
import uvicorn
import os
import sys
from pathlib import Path

# Add project root to Python path
src_root = Path(__file__).resolve().parents[2]  # /app/src
sys.path.insert(0, str(src_root))

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
    model_loaded: bool
    hopsworks_connected: bool
    timestamp: datetime
    model_info: Optional[Dict] = None
    error: Optional[str] = None

class HistoricalTrendsResponse(BaseModel):
    data_period: Dict
    hourly_patterns: Dict
    daily_patterns: Dict
    station_rankings: Dict
    demand_variability: Dict
    system_utilization: Dict
    status: str = "success"
    error: Optional[str] = None

class StationTimeSeriesResponse(BaseModel):
    time_series: Dict
    period: Dict
    status: str = "success"
    error: Optional[str] = None

# API Endpoints

@app.get("/", tags=["Info"])
def root():
    """Root endpoint with API information."""
    return {
        "service": "Velib Hourly Prediction API",
        "version": "1.0.0",
        "description": "Hourly batch predictions for all Velib stations in Paris",
        "endpoints": {
            "predictions": "/predict/all",
            "health": "/health",
            "historical_trends": "/trends/historical",
            "station_time_series": "/trends/stations",
            "model_reload": "/model/reload",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        model_loaded = prediction_service.is_model_loaded()
        hopsworks_connected = await prediction_service.check_hopsworks_connection()
        model_info = prediction_service.get_model_info()
        
        # Determine overall status
        if model_loaded and hopsworks_connected:
            status = "healthy"
        elif model_loaded:
            status = "degraded"  # Model loaded but Hopsworks unavailable
        else:
            status = "unhealthy"
        
        return HealthResponse(
            status=status,
            model_loaded=model_loaded,
            hopsworks_connected=hopsworks_connected,
            timestamp=datetime.now(),
            model_info=model_info
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            hopsworks_connected=False,
            timestamp=datetime.now(),
            error=str(e)
        )

@app.post("/predict/all", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_all_stations():
    """
    Get hourly predictions for all Velib stations.
    
    This endpoint provides batch predictions for all operational stations
    using the latest trained model and real-time station data.
    """
    try:
        logger.info("Received request for batch predictions")
        
        if not prediction_service.is_model_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded. Check /health endpoint.")
        
        # Get predictions
        predictions = await prediction_service.predict_all_stations()
        
        if not predictions:
            raise HTTPException(status_code=503, detail="Failed to generate predictions. Check model and data availability.")
        
        # Filter out None values and convert to float
        valid_predictions = {k: float(v) for k, v in predictions.items() if v is not None}
        
        return BatchPredictionResponse(
            predictions=valid_predictions,
            total_stations=len(predictions),
            successful_predictions=len(valid_predictions),
            timestamp=datetime.now(),
            model_info=prediction_service.get_model_info()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/trends/historical", response_model=HistoricalTrendsResponse, tags=["Analytics"])
async def get_historical_trends(
    days: int = Query(7, ge=1, le=30, description="Number of days to analyze (1-30)"),
    top_stations: int = Query(10, ge=5, le=50, description="Number of top stations to include (5-50)")
):
    """
    Get historical trends analysis for bike demand patterns.
    
    This endpoint provides comprehensive analysis including:
    - Hourly demand patterns across the day
    - Daily patterns (weekday vs weekend)
    - Top stations by demand and variability
    - System-wide utilization trends
    
    Args:
        days: Number of days to analyze (default: 7, max: 30)
        top_stations: Number of top stations to include in rankings (default: 10)
    """
    try:
        logger.info(f"Received request for historical trends: {days} days, {top_stations} top stations")
        
        trends_data = await prediction_service.get_historical_trends(days=days, top_stations=top_stations)
        
        if "error" in trends_data:
            raise HTTPException(status_code=503, detail=trends_data["error"])
        
        return HistoricalTrendsResponse(**trends_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting historical trends: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/trends/stations", response_model=StationTimeSeriesResponse, tags=["Analytics"])
async def get_station_time_series(
    station_names: List[str],
    days: int = Query(7, ge=1, le=30, description="Number of days to analyze (1-30)")
):
    """
    Get detailed time series data for specific stations.
    
    This endpoint provides minute-by-minute historical data for selected stations,
    including mechanical and electrical bike availability over time.
    
    Args:
        station_names: List of station names to analyze
        days: Number of days to retrieve (default: 7, max: 30)
    """
    try:
        logger.info(f"Received request for station time series: {len(station_names)} stations, {days} days")
        
        if not station_names:
            raise HTTPException(status_code=400, detail="At least one station name must be provided")
        
        if len(station_names) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 stations allowed per request")
        
        time_series_data = await prediction_service.get_station_time_series(
            station_names=station_names, 
            days=days
        )
        
        if "error" in time_series_data:
            raise HTTPException(status_code=503, detail=time_series_data["error"])
        
        return StationTimeSeriesResponse(**time_series_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting station time series: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/model/reload", tags=["Model Management"])
async def reload_model():
    """
    Reload the prediction model from the model registry.
    
    This endpoint forces a reload of the model from Hopsworks Model Registry
    or falls back to local artifacts if registry is unavailable.
    """
    try:
        logger.info("Received request to reload model")
        
        success = await prediction_service.reload_model()
        
        if success:
            return {
                "status": "success",
                "message": "Model reloaded successfully",
                "timestamp": datetime.now(),
                "model_info": prediction_service.get_model_info()
            }
        else:
            raise HTTPException(status_code=503, detail="Failed to reload model. Check logs for details.")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 