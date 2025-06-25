import hopsworks
import pandas as pd
from typing import Optional, List
from .config import HopsworksConfig
import logging
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class HopsworksFeatureStore:
    def __init__(self, config: HopsworksConfig):
        """Initialize connection to Hopsworks feature store."""
        self.config = config
        conn = hopsworks.login(
            api_key_value=self.config.api_key,
            project=self.config.project_name,
            host=self.config.host
        )
        self.fs = conn.get_feature_store()
            
    def create_feature_group(self, name: str, version: int, description: str, 
                           primary_key: List[str], online_enabled: bool = True,
                           event_time: str = "datetime"):
        """Create a new feature group in Hopsworks."""
        fg = self.fs.create_feature_group(
            name=name,
            version=version,
            description=description,
            primary_key=primary_key,
            online_enabled=online_enabled,
            event_time=event_time
        )
        return fg
    
    def get_feature_group(self, name: str, version: int):
        """Get a feature group by name and version."""
        try:
            feature_group = self.fs.get_feature_group(name=name, version=version)
            if feature_group is None:
                raise ValueError(f"Feature group {name} version {version} not found")
            return feature_group
        except Exception as e:
            raise ValueError(f"Error getting feature group {name} version {version}: {str(e)}")
    
    def get_feature_store(self):
        """Get the feature store from Hopsworks."""
        return self.fs
    
    def append_data(self, feature_group_name: str, df: pd.DataFrame, version: int = 1):
        """Append data to an existing feature group."""
        feature_group = self.fs.get_feature_group(feature_group_name, version)
        feature_group.insert(df)
        return feature_group
    
    def get_feature_view(self):
        """Get the feature view for velib data prediction."""
        try:
            return self.fs.get_feature_view(
                name=self.config.feature_view_name,
                version=1
            )
        except:
            return None
    
    def create_feature_view(self, feature_group):
        """Create a feature view from the feature group."""
        feature_view = self.fs.create_feature_view(
            name=self.config.feature_view_name,
            version=1,
            query=feature_group.select_all()
        )
        return feature_view
    
    def get_training_data(self, start_time: Optional[str] = None, end_time: Optional[str] = None) -> pd.DataFrame:
        """Get training data from the feature view."""
        feature_view = self.get_feature_view()
        if feature_view is None:
            raise ValueError("Feature view not found. Please create it first.")
        
        return feature_view.get_batch_data(
            start_time=start_time,
            end_time=end_time
        )
    
    def get_feature_vector(self, feature_vector: List[float]) -> pd.DataFrame:
        """Get feature vector for online prediction."""
        feature_view = self.get_feature_view()
        if feature_view is None:
            raise ValueError("Feature view not found. Please create it first.")
        
        return feature_view.get_feature_vector(feature_vector)
    
    def get_recent_station_data(self, hours: int = 24, feature_group_name: str = "velib_stations_status", version: int = 1) -> Optional[pd.DataFrame]:
        """
        Get recent station data for prediction purposes.
        
        Args:
            hours (int): Number of hours to look back (default: 24)
            feature_group_name (str): Name of the feature group
            version (int): Version of the feature group
            
        Returns:
            Optional[pd.DataFrame]: Station data with required columns
        """
        try:
            # Get the feature group
            feature_group = self.get_feature_group(feature_group_name, version)
            
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            logger.info(f"Querying feature group {feature_group_name} for data from {start_time} to {end_time}")
            
            # Query the feature group
            query = feature_group.select_all().filter(
                feature_group.datetime >= start_time
            ).filter(
                feature_group.datetime <= end_time
            )
            
            # Execute query
            station_data = query.read()
            
            if station_data.empty:
                logger.warning(f"No data found for the last {hours} hours")
                return None
            
            # Ensure we have the required columns
            required_columns = ['datetime', 'station_name', 'available_mechanical', 'available_electrical']
            if not all(col in station_data.columns for col in required_columns):
                logger.error(f"Missing required columns. Available: {station_data.columns.tolist()}")
                return None
            
            # Sort by datetime and station_name
            station_data = station_data.sort_values(['station_name', 'datetime'])
            
            logger.info(f"Retrieved {len(station_data)} records for {station_data['station_name'].nunique()} stations")
            return station_data
            
        except Exception as e:
            logger.error(f"Error getting recent station data: {str(e)}")
            return None

    def cleanup_old_data(self, feature_group_name: str = "velib_stations_status", version: int = 1, days_to_keep: int = 30):
        """Clean up data older than specified days from the feature group."""
        try:
            # Get the feature group
            feature_group = self.fs.get_feature_group(
                name=feature_group_name,
                version=version
            )
            
            if feature_group is None:
                raise ValueError(f"Feature group {feature_group_name} not found")
            
            # Calculate the cutoff date with timezone awareness
            cutoff_date = pd.Timestamp.now(tz='Europe/Paris') - pd.Timedelta(days=days_to_keep)
            
            # Try to get the data to delete with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Get the data to delete using proper filter syntax
                    query = feature_group.select_all()
                    query = query.filter(feature_group.datetime < cutoff_date)
                    df_to_delete = query.read()
                    break  # Success, exit retry loop
                except Exception as e:
                    if "hoodie.properties" in str(e) and attempt < max_retries - 1:
                        logger.warning(f"Hudi metadata not ready, retrying in 10 seconds (attempt {attempt + 1}/{max_retries})")
                        time.sleep(10)
                        continue
                    else:
                        raise e
            
            if not df_to_delete.empty:
                # Delete the records
                feature_group.delete(df_to_delete)
                logger.info(f"Successfully cleaned up {len(df_to_delete)} records older than {days_to_keep} days from {feature_group_name}")
            else:
                logger.info(f"No records found older than {days_to_keep} days in {feature_group_name}")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
            raise