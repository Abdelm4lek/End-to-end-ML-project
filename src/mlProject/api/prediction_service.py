"""
Simplified prediction service that reuses existing components.
"""

import asyncio
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np

from ..components.prediction import Predictor
from ..hopsworks.model_registry import HopsworksModelRegistry
from ..hopsworks.config import HopsworksConfig
from ..hopsworks.feature_store import HopsworksFeatureStore

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        """Initialize the prediction service using existing components.
        If Hopsworks is unavailable or misconfigured, gracefully fall back to a local, offline mode so that the API can still start.
        """
        try:
            self.config = HopsworksConfig()
            self.model_registry = HopsworksModelRegistry(self.config)
            self.feature_store = HopsworksFeatureStore(self.config)
        except Exception as e:
            # Any failure at this stage (missing env vars, network issues, library problems, etc.)
            # would previously crash the whole application during import time.
            # Instead, we log the problem and continue in a degraded "local only" mode.
            logger.warning(f"Hopsworks integration disabled – falling back to local mode: {e}")
            self.config = None
            self.model_registry = None
            self.feature_store = None
        
        self.predictor = None
        self.model_loaded_at = None
        self.model_version = None
        
        # Try to load model on initialization (run synchronously during startup)
        try:
            asyncio.run(self._initialize_predictor())
        except RuntimeError:
            # If we're already inside an event loop (rare, e.g. in notebooks), create a background task instead
            loop = asyncio.get_event_loop()
            loop.create_task(self._initialize_predictor())
    
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
        # If Hopsworks is disabled, immediately report False
        if not self.feature_store:
            return False
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

    async def get_historical_trends(self, days: int = 7, top_stations: int = 10) -> Dict:
        """
        Get historical trends analysis for the specified time period.
        
        Args:
            days (int): Number of days to analyze (default: 7)
            top_stations (int): Number of top stations to include in analysis
        
        Returns:
            Dict: Historical trends data with multiple analysis views
        """
        try:
            logger.info(f"Fetching historical trends for {days} days")
            
            # Get historical data
            hours = days * 24
            historical_data = self.feature_store.get_recent_station_data(hours=hours)
            
            if historical_data is None or historical_data.empty:
                logger.warning(f"No historical data found for {days} days")
                return {"error": "No historical data available"}
            
            # Calculate total bikes (mechanical + electrical)
            historical_data['total_bikes'] = (
                historical_data['available_mechanical'] + 
                historical_data['available_electrical']
            )
            
            # Convert datetime to proper format if needed
            historical_data['datetime'] = pd.to_datetime(historical_data['datetime'])
            historical_data['hour'] = historical_data['datetime'].dt.hour
            historical_data['day_of_week'] = historical_data['datetime'].dt.dayofweek
            historical_data['date'] = historical_data['datetime'].dt.date
            
            # Generate all analysis components
            trends_data = {
                "data_period": {
                    "days": days,
                    "start_date": historical_data['datetime'].min().isoformat(),
                    "end_date": historical_data['datetime'].max().isoformat(),
                    "total_records": len(historical_data),
                    "unique_stations": historical_data['station_name'].nunique()
                },
                "hourly_patterns": self._analyze_hourly_patterns(historical_data),
                "daily_patterns": self._analyze_daily_patterns(historical_data),
                "station_rankings": self._analyze_station_rankings(historical_data, top_stations),
                "demand_variability": self._analyze_demand_variability(historical_data, top_stations),
                "system_utilization": self._analyze_system_utilization(historical_data)
            }
            
            logger.info(f"Generated historical trends analysis for {len(historical_data)} records")
            return trends_data
            
        except Exception as e:
            logger.error(f"Error getting historical trends: {str(e)}")
            return {"error": f"Failed to get historical trends: {str(e)}"}
    
    def _analyze_hourly_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze hourly demand patterns across all stations."""
        hourly_avg = df.groupby('hour')['total_bikes'].agg(['mean', 'std', 'count']).round(2)
        
        return {
            "hourly_averages": hourly_avg['mean'].to_dict(),
            "hourly_std": hourly_avg['std'].to_dict(),
            "peak_hour": int(hourly_avg['mean'].idxmax()),
            "low_hour": int(hourly_avg['mean'].idxmin()),
            "peak_demand": float(hourly_avg['mean'].max()),
            "low_demand": float(hourly_avg['mean'].min())
        }
    
    def _analyze_daily_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze daily demand patterns (weekday vs weekend)."""
        daily_avg = df.groupby('day_of_week')['total_bikes'].agg(['mean', 'std']).round(2)
        
        # Calculate weekday vs weekend averages
        weekday_avg = daily_avg.loc[0:4, 'mean'].mean()  # Mon-Fri
        weekend_avg = daily_avg.loc[5:6, 'mean'].mean()  # Sat-Sun
        
        return {
            "daily_averages": daily_avg['mean'].to_dict(),
            "daily_std": daily_avg['std'].to_dict(),
            "weekday_average": round(weekday_avg, 2),
            "weekend_average": round(weekend_avg, 2),
            "weekend_vs_weekday_ratio": round(weekend_avg / weekday_avg, 2) if weekday_avg > 0 else 0
        }
    
    def _analyze_station_rankings(self, df: pd.DataFrame, top_n: int) -> Dict:
        """Analyze station rankings by various metrics."""
        station_stats = df.groupby('station_name')['total_bikes'].agg([
            'mean', 'max', 'min', 'std', 'count'
        ]).round(2)
        
        # Top stations by average demand
        top_by_avg = station_stats.nlargest(top_n, 'mean')
        
        # Most variable stations (highest std dev)
        top_by_variability = station_stats.nlargest(top_n, 'std')
        
        return {
            "top_by_average_demand": {
                "stations": top_by_avg.index.tolist(),
                "average_bikes": top_by_avg['mean'].tolist(),
                "max_bikes": top_by_avg['max'].tolist()
            },
            "most_variable_stations": {
                "stations": top_by_variability.index.tolist(),
                "std_dev": top_by_variability['std'].tolist(),
                "average_bikes": top_by_variability['mean'].tolist()
            }
        }
    
    def _analyze_demand_variability(self, df: pd.DataFrame, top_n: int) -> Dict:
        """Analyze demand variability and capacity utilization patterns."""
        # Group by station and calculate daily patterns
        station_daily = df.groupby(['station_name', 'date'])['total_bikes'].agg(['mean', 'max', 'min']).reset_index()
        
        # Calculate variability metrics per station
        variability_stats = station_daily.groupby('station_name').agg({
            'mean': ['mean', 'std'],
            'max': 'mean',
            'min': 'mean'
        }).round(2)
        
        # Flatten column names
        variability_stats.columns = ['avg_daily_mean', 'daily_variability', 'avg_daily_max', 'avg_daily_min']
        
        # Get top variable stations
        top_variable = variability_stats.nlargest(top_n, 'daily_variability')
        
        return {
            "most_consistent_stations": variability_stats.nsmallest(5, 'daily_variability').index.tolist(),
            "most_variable_stations": top_variable.index.tolist(),
            "variability_scores": top_variable['daily_variability'].tolist(),
            "system_wide_variability": float(variability_stats['daily_variability'].mean())
        }
    
    def _analyze_system_utilization(self, df: pd.DataFrame) -> Dict:
        """Analyze overall system utilization patterns."""
        total_bikes_by_hour = df.groupby(['datetime'])['total_bikes'].sum()
        
        # Calculate utilization trends
        daily_totals = df.groupby('date')['total_bikes'].sum()
        
        return {
            "average_system_bikes": float(total_bikes_by_hour.mean()),
            "peak_system_bikes": float(total_bikes_by_hour.max()),
            "min_system_bikes": float(total_bikes_by_hour.min()),
            "daily_trend": {
                "dates": [str(date) for date in daily_totals.index],
                "total_bikes": daily_totals.tolist()
            },
            "utilization_stability": float(1 - (total_bikes_by_hour.std() / total_bikes_by_hour.mean())) if total_bikes_by_hour.mean() > 0 else 0
        }

    async def get_station_time_series(self, station_names: List[str], days: int = 7) -> Dict:
        """
        Get detailed time series data for specific stations.
        
        Args:
            station_names (List[str]): List of station names to analyze
            days (int): Number of days to retrieve
        
        Returns:
            Dict: Time series data for the specified stations
        """
        try:
            logger.info(f"Fetching time series for {len(station_names)} stations over {days} days")
            
            # Get historical data
            hours = days * 24
            historical_data = self.feature_store.get_recent_station_data(hours=hours)
            
            if historical_data is None or historical_data.empty:
                return {"error": "No historical data available"}
            
            # Filter for requested stations
            station_data = historical_data[historical_data['station_name'].isin(station_names)].copy()
            
            if station_data.empty:
                return {"error": "No data found for requested stations"}
            
            # Calculate total bikes
            station_data['total_bikes'] = (
                station_data['available_mechanical'] + 
                station_data['available_electrical']
            )
            
            # Prepare time series data
            time_series = {}
            for station in station_names:
                station_subset = station_data[station_data['station_name'] == station].copy()
                station_subset = station_subset.sort_values('datetime')
                
                time_series[station] = {
                    "timestamps": [dt.isoformat() for dt in station_subset['datetime']],
                    "total_bikes": station_subset['total_bikes'].tolist(),
                    "mechanical_bikes": station_subset['available_mechanical'].tolist(),
                    "electrical_bikes": station_subset['available_electrical'].tolist(),
                    "data_points": len(station_subset)
                }
            
            return {
                "time_series": time_series,
                "period": {
                    "days": days,
                    "start_date": historical_data['datetime'].min().isoformat(),
                    "end_date": historical_data['datetime'].max().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting station time series: {str(e)}")
            return {"error": f"Failed to get station time series: {str(e)}"} 