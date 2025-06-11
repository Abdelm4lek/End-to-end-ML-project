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
            stations_df['station_id'] = stations_df['station_id'].astype(str)
            status_df['station_id'] = status_df['station_id'].astype(str)
            
            logger.info(f"Number of stations in stations info: {len(stations_df)}")
            logger.info(f"Number of stations in stations status: {len(status_df)}")
            
            # Check for missing stations
            missing_stations = set(status_df['station_id']) - set(stations_df['station_id'])
            if missing_stations:
                logger.warning(f"Found {len(missing_stations)} stations in status that are not in station info")
                # Filter out stations that don't exist in station info
                status_df = status_df[status_df['station_id'].isin(stations_df['station_id'])]
            
            # Merge station info and status
            merged_df = pd.merge(
                stations_df,
                status_df,
                on='station_id',
                how='inner',  
                suffixes=('', '_status')
            )
            
            logger.info(f"Number of stations after merge: {len(merged_df)}")
            
            merged_df['last_reported'] = pd.to_datetime(merged_df['last_reported'], unit='s')
            
            # Create the new structure
            processed_df = pd.DataFrame({
                'datetime': merged_df['last_reported'], 
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
            
            logger.info(f"Number of unique station names in processed data: {processed_df['station_name'].nunique()}")
            
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
            stations_df = pd.DataFrame(data['data']['stations']).astype({'station_id': str})
            
            logger.info(f"Number of stations in API response: {len(stations_df)}")
            
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
            # fetch and store station information
            stations_df = self.fetch_station_info()
            if stations_df is None:
                logger.error("Failed to fetch station information")
                return
            
            # Verify station information was stored
            conn = self.db._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM stations")
                station_count = cursor.fetchone()[0]
                logger.info(f"Verified {station_count} stations in database")
            finally:
                self.db._release_connection(conn)
            
            # fetch station status
            status_df = self.fetch_station_status()
            if status_df is None:
                logger.error("Failed to fetch station status")
                return
            
            # Preprocess the data
            processed_df = self.preprocess_station_data(stations_df, status_df)
            
            # Verify all station names exist in the stations table
            conn = self.db._get_connection()
            try:
                cursor = conn.cursor()
                for station_name in processed_df['station_name'].unique():
                    cursor.execute("SELECT COUNT(*) FROM stations WHERE name = %s", (station_name,))
                    if cursor.fetchone()[0] == 0:
                        logger.error(f"Station name '{station_name}' not found in stations table")
                        return
            finally:
                self.db._release_connection(conn)
            
            # Store the processed data
            self.db.store_hourly_observations(processed_df)
            logger.info("Successfully stored processed hourly observations")
            
            # Log total number of rows in hourly_observations
            conn = self.db._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM hourly_observations")
                total_rows = cursor.fetchone()[0]
                logger.info(f"Total number of rows in hourly_observations table: {total_rows}")
            finally:
                self.db._release_connection(conn)
            
            # Clean up data older than 30 days
            self.db.cleanup_old_data(days_to_keep=30)

        except Exception as e:
            logger.error(f"Error in data collection: {str(e)}")
            raise

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