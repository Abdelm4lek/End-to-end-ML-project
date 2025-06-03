import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from src.data.database import VelibDatabase

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self, model_path: str):
        """
        Initialize the prediction pipeline with a trained model
        
        Args:
            model_path (str): Path to the trained model file
        """
        self.model = joblib.load(model_path)
        self.db = VelibDatabase()
        logger.info(f"Loaded model from {model_path}")
    
    def prepare_station_features(self, station_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for a single station using the last 24 hours of data
        
        Args:
            station_data (pd.DataFrame): DataFrame containing station data with columns:
                - datetime: timestamp
                - available_mechanical: number of available mechanical bikes
                - available_electrical: number of available electrical bikes
                - station_name: name of the station
                
        Returns:
            pd.DataFrame: DataFrame with prepared features
        """
        # Calculate total available bikes
        station_data['total_available'] = station_data['available_mechanical'] + station_data['available_electrical']
        
        # Sort by datetime to ensure proper lag calculation
        station_data = station_data.sort_values('datetime')
        
        # Create lagged features (last 24 hours)
        for i in range(1, 25):
            station_data[f'total_available_lag_{i}'] = station_data['total_available'].shift(i)
        
        # Drop rows with NaN values (first 24 hours)
        features_df = station_data.dropna()
        
        if len(features_df) == 0:
            logger.warning("No valid features could be created for this station")
            return pd.DataFrame()
        
        # Select only the feature columns
        feature_columns = [f'total_available_lag_{i}' for i in range(1, 25)]
        return features_df[feature_columns]
    
    def predict_station(self, station_data: pd.DataFrame) -> Optional[float]:
        """
        Make prediction for a single station
        
        Args:
            station_data (pd.DataFrame): DataFrame containing station data
            
        Returns:
            Optional[float]: Predicted number of bikes for the next hour, or None if prediction cannot be made
        """
        try:
            # Prepare features
            features = self.prepare_station_features(station_data)
            
            if features.empty:
                return None
            
            # Make prediction using the last row of features
            prediction = self.model.predict(features.iloc[-1:])[0]
            
            # Ensure prediction is non-negative and rounded to nearest integer
            prediction = max(0, round(prediction))
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error making prediction for station: {str(e)}")
            return None
    
    def predict_all_stations(self) -> Dict[str, float]:
        """
        Make predictions for all stations using data from the database
        
        Returns:
            Dict[str, float]: Dictionary mapping station names to their predictions
        """
        try:
            # Get last 24 hours of data for all stations
            stations_data = self.db.get_all_stations_last_24h()
            
            if stations_data is None or stations_data.empty:
                logger.error("No data available from database")
                return {}
            
            predictions = {}
            
            for station_name in stations_data['station_name'].unique():
                # Get data for this station
                station_df = stations_data[stations_data['station_name'] == station_name]
                
                # Make prediction
                prediction = self.predict_station(station_df)
                
                if prediction is not None:
                    predictions[station_name] = prediction
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions for all stations: {str(e)}")
            return {}
    
    def get_prediction_timeframe(self) -> tuple:
        """
        Get the start and end times for the prediction window
        
        Returns:
            tuple: (start_time, end_time) as datetime objects
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        return start_time, end_time 