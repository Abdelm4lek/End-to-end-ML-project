import time
import schedule
import requests
import pandas as pd
from datetime import datetime
from data.database import VelibDatabase
import logging
import json
import os

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VelibDataCollector:
    def __init__(self):
        self.db = VelibDatabase()
        self.station_info_url = "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_information.json"
        self.station_status_url = "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_status.json"
    
    def preprocess_station_data(self, stations_df, status_df):
        """Transform raw data into desired structure"""
        try:
            # Ensure both DataFrames have station_id as string
            stations_df['station_id'] = stations_df['station_id'].astype(str)
            status_df['station_id'] = status_df['station_id'].astype(str)
            
            # Merge station info and status
            merged_df = pd.merge(
                stations_df,
                status_df,
                on='station_id',
                how='left',
                suffixes=('', '_status')
            )
            
            # Create the new structure
            processed_df = pd.DataFrame({
                'datetime': datetime.now(),
                'capacity': merged_df['capacity'],
                'available_mechanical': merged_df['num_bikes_available_types'].apply(
                    lambda x: x[0]['mechanical'] if isinstance(x, list) and len(x) > 0 else 0
                ),
                'available_electrical': merged_df['num_bikes_available_types'].apply(
                    lambda x: x[1]['ebike'] if isinstance(x, list) and len(x) > 1 else 0
                ),
                'station_name': merged_df['name'],
                'station_geo': merged_df.apply(
                    lambda x: json.dumps({'lat': x['lat'], 'lon': x['lon']}),
                    axis=1
                ),
                'operative': merged_df['is_renting'] & merged_df['is_installed']
            })
            
            # Ensure all columns have the correct data types
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
    
    def fetch_station_info(self):
        """Fetch and store station information"""
        try:
            response = requests.get(self.station_info_url)
            data = response.json()
            # Convert station_id to string when creating DataFrame
            stations_df = pd.DataFrame(data['data']['stations']).astype({'station_id': str})
            self.db.store_station_info(stations_df)
            logger.info("Successfully updated station information")
            return stations_df
        except Exception as e:
            logger.error(f"Error fetching station info: {str(e)}")
            return None
    
    def fetch_station_status(self):
        """Fetch and store current station status"""
        try:
            response = requests.get(self.station_status_url)
            data = response.json()
            # Convert station_id to string when creating DataFrame
            status_df = pd.DataFrame(data['data']['stations']).astype({'station_id': str})
            return status_df
        except Exception as e:
            logger.error(f"Error fetching station status: {str(e)}")
            return None
    
    def collect_data(self):
        """Collect both station info and status"""
        try:
            # Fetch both dataframes
            stations_df = self.fetch_station_info()
            status_df = self.fetch_station_status()
            
            if stations_df is not None and status_df is not None:
                # Preprocess the data
                processed_df = self.preprocess_station_data(stations_df, status_df)
                
                # Store the processed data
                self.db.store_hourly_observations(processed_df)
                logger.info("Successfully stored processed hourly observations")
            
            # Clean up data older than 30 days
            self.db.cleanup_old_data(days_to_keep=30)

        except Exception as e:
            logger.error(f"Error in data collection: {str(e)}")

def main():
    collector = VelibDataCollector()
    
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