import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VelibDatabase:
    def __init__(self, db_path='velib_data.db'):
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create stations table for static information
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stations (
                    station_id INTEGER PRIMARY KEY,
                    name TEXT,
                    lat REAL,
                    lon REAL,
                    capacity INTEGER,
                    last_updated TIMESTAMP
                )
            ''')
            
            # Create hourly observations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS hourly_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    station_id INTEGER,
                    timestamp TIMESTAMP,
                    num_bikes_available INTEGER,
                    num_docks_available INTEGER,
                    is_renting BOOLEAN,
                    FOREIGN KEY (station_id) REFERENCES stations(station_id)
                )
            ''')
            
            conn.commit()
    
    def store_station_info(self, stations_df):
        """Store or update station information"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                stations_df['last_updated'] = datetime.now()
                stations_df.to_sql('stations', conn, if_exists='replace', index=False)
                logger.info(f"Stored information for {len(stations_df)} stations")
        except Exception as e:
            logger.error(f"Error storing station info: {str(e)}")
    
    def store_hourly_observations(self, observations_df):
        """Store hourly observations for all stations"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                observations_df['timestamp'] = datetime.now()
                observations_df.to_sql('hourly_observations', conn, if_exists='append', index=False)
                logger.info(f"Stored hourly observations for {len(observations_df)} stations")
        except Exception as e:
            logger.error(f"Error storing hourly observations: {str(e)}")
    
    def get_last_24h_data(self, station_id):
        """Retrieve last 24 hours of data for a specific station"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                SELECT h.*, s.name, s.lat, s.lon, s.capacity
                FROM hourly_observations h
                JOIN stations s ON h.station_id = s.station_id
                WHERE h.station_id = ?
                AND h.timestamp >= datetime('now', '-24 hours')
                ORDER BY h.timestamp ASC
                """
                df = pd.read_sql_query(query, conn, params=(station_id,))
                return df
        except Exception as e:
            logger.error(f"Error retrieving data for station {station_id}: {str(e)}")
            return None
    
    def get_all_stations_last_24h(self):
        """Retrieve last 24 hours of data for all stations"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                SELECT h.*, s.name, s.lat, s.lon, s.capacity
                FROM hourly_observations h
                JOIN stations s ON h.station_id = s.station_id
                WHERE h.timestamp >= datetime('now', '-24 hours')
                ORDER BY h.station_id, h.timestamp ASC
                """
                df = pd.read_sql_query(query, conn)
                return df
        except Exception as e:
            logger.error(f"Error retrieving all stations data: {str(e)}")
            return None
    
    def cleanup_old_data(self, days_to_keep=30):
        """Remove data older than specified days"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM hourly_observations 
                    WHERE timestamp < datetime('now', '-? days')
                """, (days_to_keep,))
                conn.commit()
                logger.info(f"Cleaned up data older than {days_to_keep} days")
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}") 