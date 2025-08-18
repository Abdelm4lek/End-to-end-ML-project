import pandas as pd
import polars as pl
import numpy as np
import joblib
import logging
import os
from typing import Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path
from src.mlProject.hopsworks.feature_store import HopsworksFeatureStore
from src.mlProject.hopsworks.config import HopsworksConfig
from src.mlProject.hopsworks.feature_schema import get_feature_group_schema
from src.mlProject.utils.common import create_dirs

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self, model_path: str):
        """
        Initialize the prediction pipeline with a trained model
        
        Args:
            model_path (str): Path to the trained model file
        """
        self.model = joblib.load(model_path)
        
        # Initialize Hopsworks connection
        try:
            self.config = HopsworksConfig()
            self.feature_store = HopsworksFeatureStore(self.config)
            
            # Get feature group
            schema = get_feature_group_schema()
            self.feature_group = self.feature_store.get_feature_group(
                name=schema["name"],
                version=schema["version"]
            )
            logger.info("Connected to Hopsworks Feature Store")
        except Exception as e:
            logger.error(f"Failed to connect to Hopsworks: {str(e)}")
            raise ConnectionError(f"Failed to connect to Hopsworks Feature Store: {str(e)}")
        
        logger.info(f"Loaded model from {model_path}")
    
    def get_all_stations_last_24h(self) -> Optional[pd.DataFrame]:
        """
        Get last 24 hours of data for all stations from feature store.
        
        Returns:
            Optional[pd.DataFrame]: Station data with columns:
                - datetime, station_name, available_mechanical, available_electrical
        """
        return self.feature_store.get_recent_station_data(hours=24)
    
    def preprocess_prediction_data(self, stations_data: pd.DataFrame) -> pl.DataFrame:
        """
        Preprocess prediction data using the same logic as in data transformation.
        Adapted from DataTransformation.preprocess_data() method.
        
        Args:
            stations_data (pd.DataFrame): Raw station data from Hopsworks
            
        Returns:
            pl.DataFrame: Preprocessed data ready for feature creation
        """
        # Convert to polars for processing (same as in training)
        df = pl.from_pandas(stations_data)
        
        # Calculate total available bikes (same as training)
        df = df.with_columns(
            total_available = pl.col("available_mechanical") + pl.col("available_electrical")
        )
        
        # Parse datetime and extract components (same as training)
        df = df.with_columns(
            datetime = pl.col("datetime").str.to_datetime() if df["datetime"].dtype == pl.Utf8 else pl.col("datetime")
        ).with_columns(
            date = pl.col("datetime").dt.date(),
            hour = pl.col("datetime").dt.hour().cast(pl.Int32)
        )
        
        # Group by date, hour, and station, taking last value in each hour (same as training)
        df = df.group_by(["date", "hour", "station_name"]).agg([
            pl.col("total_available").last(),
            pl.col("available_mechanical").last(), 
            pl.col("available_electrical").last()
        ])
        
        # Select and sort columns (same structure as training)
        df = df.select([
            "date",
            "hour", 
            "station_name",
            "total_available",
            "available_mechanical",
            "available_electrical"
        ]).sort(["date", "hour", "station_name"])
        
        logger.info(f"Preprocessed prediction data. Shape: {df.shape}")
        return df
    
    def create_station_mapping(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Creates a station mapping exactly like in training.
        Adapted from DataTransformation.create_station_mapping() method.
        """
        station_mapping = (
            df.sort("station_name")
            .select("station_name")
            .unique()
            .with_row_index("station_id")
        )
        return station_mapping
    
    def create_lagged_features_for_prediction(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Creates lagged features for prediction using the exact same logic as training.
        Adapted from DataTransformation.create_lagged_features() method.
        
        Args:
            df (pl.DataFrame): Preprocessed station data
            
        Returns:
            pl.DataFrame: DataFrame with lagged features ready for prediction
        """
        station_mapping = self.create_station_mapping(df)

        # Add station IDs to original dataframe by joining on station_name (same as training)
        new_df = df.join(
            station_mapping,
            on="station_name",
            how="left"
        )

        # Keep station_id as numeric (int64) to match training data
        new_df = new_df.with_columns(
            pl.col("station_id").cast(pl.Int64)
        )

        # Select and reorder columns (same as training)
        new_df = new_df.select([
            "station_id",
            "date",
            "hour",
            "total_available",
            "available_mechanical", 
            "available_electrical"
        ])
        
        # Sort by station_id, date and hour (same as training)
        new_df = new_df.sort(["station_id", "date", "hour"])

        # Get unique station IDs
        unique_stations = new_df.get_column("station_id").unique().to_list()

        # Create list to store DataFrames with lagged features for each station
        station_lag_dfs = []

        # For each station, create lagged features (same logic as training)
        for station_id in unique_stations:
            # Filter data for current station
            station_df = new_df.filter(pl.col("station_id") == station_id)
            
            # Create lag columns for total_available (same as training)
            lag_columns = []
            for i in range(1, 25):
                lag_columns.append(
                    pl.col("total_available").shift(i).alias(f"total_available_lag_{i}")
                )
            
            # Add lag columns to station DataFrame (same structure as training)
            station_with_lags = station_df.with_columns(lag_columns).select(
                ["station_id", "date", "hour", "total_available"] + 
                [f"total_available_lag_{i}" for i in range(1, 25)]
            )
            
            # For prediction, don't drop all nulls - just get the most recent row with some lags
            # We need at least 12 valid lag features to make a prediction
            if len(station_with_lags) > 0:
                lag_cols = [f"total_available_lag_{i}" for i in range(1, 25)]
                station_with_lags = station_with_lags.with_columns(
                    pl.sum_horizontal([pl.col(col).is_not_null().cast(pl.Int32) for col in lag_cols]).alias("valid_lag_count")
                )
                
                # Filter for rows with at least 12 valid lags
                valid_rows = station_with_lags.filter(pl.col("valid_lag_count") >= 12)
                if len(valid_rows) > 0:
                    # Take the most recent valid row
                    latest_row = valid_rows.tail(1)
                    station_lag_dfs.append(latest_row)
        
        if len(station_lag_dfs) == 0:
            logger.warning("No stations have sufficient lag features for prediction")
            return pl.DataFrame()
        
        # Concatenate all station DataFrames (same as training)
        lags_df = pl.concat(station_lag_dfs).sort(["date", "hour"])

        logger.info(f"Created lagged features for prediction. Shape: {lags_df.shape}")
        return lags_df
    
    def prepare_all_stations_features(self, stations_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ALL stations using the exact same pipeline as training.
        This ensures consistency and avoids mismatches.
        
        Args:
            stations_data (pd.DataFrame): Raw station data from Hopsworks
            
        Returns:
            pd.DataFrame: Features ready for model prediction, indexed by station_name
        """
        logger.info(f"Preparing features for {stations_data['station_name'].nunique()} stations...")
        
        # Step 1: Preprocess data (same as training)
        preprocessed_df = self.preprocess_prediction_data(stations_data)
        
        # Step 2: Create lagged features (same as training) 
        features_df = self.create_lagged_features_for_prediction(preprocessed_df)
        
        if features_df.is_empty():
            logger.error("No valid features could be created for any station")
            return pd.DataFrame()
        
        # Convert back to pandas for model prediction
        features_pd = features_df.to_pandas()
        
        # Create a mapping from station_id back to station_name for indexing
        station_mapping = preprocessed_df.select(["station_name"]).unique().with_row_index("station_id").to_pandas()
        station_id_to_name = dict(zip(station_mapping['station_id'], station_mapping['station_name']))
        
        # Add station_name for indexing
        features_pd['station_name'] = features_pd['station_id'].map(station_id_to_name)
        
        # Select features in the same order as training: station_id, hour, lag_1 to lag_24
        # Drop date and total_available (target) just like in training
        feature_columns = ['station_id', 'hour'] + [f'total_available_lag_{i}' for i in range(1, 25)]
        final_features = features_pd[['station_name'] + feature_columns].set_index('station_name')
        
        # Fill any NaN lag values with 0
        lag_cols = [f'total_available_lag_{i}' for i in range(1, 25)]
        final_features[lag_cols] = final_features[lag_cols].fillna(0)
        
        logger.info(f"Feature preparation completed. Shape: {final_features.shape}")
        
        return final_features
    
    def prepare_station_features(self, station_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for a single station using the last 24 hours of data
        (Kept for backward compatibility, but recommend using prepare_all_stations_features for efficiency)
        
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
        Make predictions for all stations using data from Hopsworks Feature Store
        
        Returns:
            Dict[str, float]: Dictionary mapping station names to their predictions
        """
        try:
            # Get last 24 hours of data for all stations from Hopsworks
            stations_data = self.get_all_stations_last_24h()
            
            if stations_data is None or stations_data.empty:
                logger.error("No data available from Hopsworks Feature Store")
                return {}
            
            # Prepare features for ALL stations at once (much more efficient)
            features = self.prepare_all_stations_features(stations_data)
            
            if features.empty:
                logger.error("No valid features could be prepared for any station")
                return {}
            
            predictions = {}
            station_count = len(features)
            logger.info(f"Making predictions for {station_count} stations...")
            
            # Make predictions for each station using the prepared features
            for station_name in features.index:
                try:
                    # Get features for this station
                    station_features = features.loc[station_name:station_name]
                    
                    # Make prediction
                    prediction = self.model.predict(station_features)[0]
                    
                    # Ensure prediction is non-negative and rounded to nearest integer
                    prediction = max(0, round(prediction))
                    
                    predictions[station_name] = prediction
                    logger.debug(f"Prediction for {station_name}: {prediction}")
                    
                except Exception as e:
                    logger.warning(f"Could not make prediction for station {station_name}: {str(e)}")
            
            logger.info(f"Successfully made predictions for {len(predictions)} out of {station_count} stations")
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
    
    def save_predictions_to_csv(self, predictions: Dict[str, float], output_path: str) -> None:
        """
        Save predictions to a CSV file in the format (station_name, predicted_total_available)
        
        Args:
            predictions (Dict[str, float]): Dictionary mapping station names to predictions
            output_path (str): Path where to save the CSV file
        """
        try:
            # Create the directory if it doesn't exist
            create_dirs([Path(output_path).parent])
            
            # Convert predictions to DataFrame
            predictions_df = pd.DataFrame([
                {"station_name": station_name, "predicted_total_available": int(prediction)}
                for station_name, prediction in predictions.items()
            ])
            
            # Add timestamp for when predictions were made
            predictions_df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Sort by station name for consistency
            predictions_df = predictions_df[["timestamp","station_name", "predicted_total_available"]]
            predictions_df = predictions_df.sort_values("station_name")
            
            # Check if file exists and append or create new
            if Path(output_path).exists():
                predictions_df.to_csv(output_path, mode='a', header=False, index=False)
                logger.info(f"Appended {len(predictions)} predictions to existing file {output_path}")
            else:
                predictions_df.to_csv(output_path, index=False)
                logger.info(f"Created new file and saved {len(predictions)} predictions to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving predictions to CSV: {str(e)}")
            raise e 