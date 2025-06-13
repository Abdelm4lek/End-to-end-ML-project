import time
import schedule
import requests
import pandas as pd
from datetime import datetime
import logging
import json
from typing import Optional, Tuple
from .feature_store import HopsworksFeatureStore
from .feature_schema import get_feature_group_schema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VelibHopsworksCollector:
    def __init__(self):
        """Initialize the Velib data collector with Hopsworks feature store."""
        self.feature_store = HopsworksFeatureStore()
        self.station_info_url = "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_information.json"
        self.station_status_url = "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_status.json"
        
        # Get or create feature group
        schema = get_feature_group_schema()
        try:
            self.feature_group = self.feature_store.get_feature_group(
                name=schema["name"],
                version=schema["version"]
            )
        except:
            self.feature_group = self.feature_store.create_feature_group(
                name=schema["name"],
                version=schema["version"],
                description=schema["description"],
                primary_key=schema["primary_key"],
                online_enabled=schema["online_enabled"],
                event_time=schema["event_time"]
            )
    
    def fetch_station_info(self) -> Optional[pd.DataFrame]:
        """Fetch station information from Velib API."""
        try:
            response = requests.get(self.station_info_url)
            data = response.json()
            stations_df = pd.DataFrame(data['data']['stations'])
            stations_df['station_id'] = stations_df['station_id'].astype(str)
            logger.info(f"Fetched information for {len(stations_df)} stations")
            return stations_df
        except Exception as e:
            logger.error(f"Error fetching station info: {str(e)}")
            return None
    
    def fetch_station_status(self) -> Optional[pd.DataFrame]:
        """Fetch current station status from Velib API."""
        try:
            response = requests.get(self.station_status_url)
            data = response.json()
            status_df = pd.DataFrame(data['data']['stations'])
            status_df['station_id'] = status_df['station_id'].astype(str)
            logger.info(f"Fetched status for {len(status_df)} stations")
            return status_df
        except Exception as e:
            logger.error(f"Error fetching station status: {str(e)}")
            return None
    
    def preprocess_data(self, stations_df: pd.DataFrame, status_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess and merge station information and status data."""
        try:
            # Merge station info and status
            merged_df = pd.merge(
                stations_df,
                status_df,
                on='station_id',
                how='inner',
                suffixes=('', '_status')
            )
            
            # Extract mechanical and electrical bike counts
            merged_df['available_mechanical'] = merged_df['num_bikes_available_types'].apply(
                lambda x: x[0]['mechanical'] if isinstance(x, list) and len(x) > 0 else 0
            )
            merged_df['available_electrical'] = merged_df['num_bikes_available_types'].apply(
                lambda x: x[1]['ebike'] if isinstance(x, list) and len(x) > 1 else 0
            )
            
            # Convert timestamps
            merged_df['datetime'] = pd.to_datetime(merged_df['last_reported'], unit='s')
            
            # Create station_geo JSON
            merged_df['station_geo'] = merged_df.apply(
                lambda x: json.dumps({'lat': x['lat'], 'lon': x['lon']}),
                axis=1
            )
            
            # Create operative status
            merged_df['operative'] = merged_df['is_renting'] & merged_df['is_installed']
            
            # Select and rename columns to match schema
            processed_df = merged_df[[
                'datetime', 'capacity', 'available_mechanical',
                'available_electrical', 'name', 'station_geo', 'operative'
            ]].rename(columns={'name': 'station_name'})
            
            # Ensure correct data types
            processed_df = processed_df.astype({
                'capacity': 'int32',
                'available_mechanical': 'int32',
                'available_electrical': 'int32',
                'station_name': 'str',
                'station_geo': 'str',
                'operative': 'bool'
            })
            
            return processed_df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def collect_data(self):
        """Collect and store Velib data in Hopsworks feature store."""
        try:
            # Fetch data
            stations_df = self.fetch_station_info()
            if stations_df is None:
                logger.error("Failed to fetch station information")
                return
            
            status_df = self.fetch_station_status()
            if status_df is None:
                logger.error("Failed to fetch station status")
                return
            
            # Preprocess data
            processed_df = self.preprocess_data(stations_df, status_df)
            
            # Append to feature group
            self.feature_group.insert(processed_df)
            logger.info(f"Successfully stored data for {len(processed_df)} stations")
            
        except Exception as e:
            logger.error(f"Error in data collection: {str(e)}")
            raise

def main():
    """Main function to run the data collector."""
    collector = VelibHopsworksCollector()
    
    # Schedule data collection every hour
    schedule.every().hour.at(":00").do(collector.collect_data)
    
    # Initial data collection
    collector.collect_data()
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60) 

if __name__ == "__main__":
    main()