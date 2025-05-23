import time
import schedule
import requests
import pandas as pd
from datetime import datetime
from data.database import VelibDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VelibDataCollector:
    def __init__(self):
        self.db = VelibDatabase()
        self.station_info_url = "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_information.json"
        self.station_status_url = "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_status.json"
    
    def fetch_station_info(self):
        """Fetch and store station information"""
        try:
            response = requests.get(self.station_info_url)
            data = response.json()
            stations_df = pd.DataFrame(data['data']['stations'])
            self.db.store_station_info(stations_df)
            logger.info("Successfully updated station information")
        except Exception as e:
            logger.error(f"Error fetching station info: {str(e)}")
    
    def fetch_station_status(self):
        """Fetch and store current station status"""
        try:
            response = requests.get(self.station_status_url)
            data = response.json()
            status_df = pd.DataFrame(data['data']['stations'])
            self.db.store_hourly_observations(status_df)
            logger.info("Successfully stored hourly observations")
        except Exception as e:
            logger.error(f"Error fetching station status: {str(e)}")
    
    def collect_data(self):
        """Collect both station info and status"""
        self.fetch_station_info()
        self.fetch_station_status()
        # Clean up data older than 30 days
        self.db.cleanup_old_data(days_to_keep=30)

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