import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import logging
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2 import pool

# Load environment variables
load_dotenv('DB_credentials.env')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VelibDatabase:
    def __init__(self, db_type=None):
        self.db_type = db_type or os.getenv('DB_TYPE', 'sqlite')
        self._validate_environment()
        
        if self.db_type == 'postgres':
            self._init_postgres_pool()
        else:
            self.db_path = os.getenv('SQLITE_DB_PATH', 'velib_data.db')
            self._create_tables()
    
    def _validate_environment(self):
        """Validate that all required environment variables are set"""
        if self.db_type == 'postgres':
            required_vars = ['DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                raise ValueError(
                    f"Missing required environment variables for PostgreSQL: {', '.join(missing_vars)}. "
                    "Please set these variables in your environment or .env file."
                )
    
    def _init_postgres_pool(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=os.getenv('DB_HOST'),
                database=os.getenv('DB_NAME'),
                user=os.getenv('DB_USER'),
                password=os.getenv('DB_PASSWORD'),
                port=os.getenv('DB_PORT', '5432'),
                sslmode='require' if os.getenv('DB_SSL', 'true').lower() == 'true' else 'disable'
            )
            self._create_tables()
            logger.info("Successfully initialized PostgreSQL connection pool")
        except Exception as e:
            logger.error(f"Error initializing PostgreSQL pool: {str(e)}")
            raise
    
    def _get_connection(self):
        """Get a database connection based on the configured type"""
        if self.db_type == 'postgres':
            return self.pool.getconn()
        else:
            return sqlite3.connect(self.db_path)
    
    def _release_connection(self, conn):
        """Release a database connection"""
        if self.db_type == 'postgres':
            self.pool.putconn(conn)
        else:
            conn.close()
    
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            if self.db_type == 'postgres':
                # Drop existing tables if they exist
                cursor.execute('DROP TABLE IF EXISTS hourly_observations CASCADE')
                cursor.execute('DROP TABLE IF EXISTS stations CASCADE')
                
                # Create stations table for static information
                cursor.execute('''
                    CREATE TABLE stations (
                        name TEXT PRIMARY KEY,
                        lat REAL,
                        lon REAL,
                        capacity INTEGER,
                        last_updated TIMESTAMP
                    )
                ''')
                
                # Create hourly observations table with new structure
                cursor.execute('''
                    CREATE TABLE hourly_observations (
                        id SERIAL PRIMARY KEY,
                        datetime TIMESTAMP,
                        capacity INTEGER,
                        available_mechanical INTEGER,
                        available_electrical INTEGER,
                        station_name TEXT REFERENCES stations(name),
                        station_geo JSONB,
                        operative BOOLEAN
                    )
                ''')
            else:
                # SQLite tables
                cursor.execute('DROP TABLE IF EXISTS hourly_observations')
                cursor.execute('DROP TABLE IF EXISTS stations')
                
                cursor.execute('''
                    CREATE TABLE stations (
                        name TEXT PRIMARY KEY,
                        lat REAL,
                        lon REAL,
                        capacity INTEGER,
                        last_updated TIMESTAMP
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE hourly_observations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        datetime TIMESTAMP,
                        capacity INTEGER,
                        available_mechanical INTEGER,
                        available_electrical INTEGER,
                        station_name TEXT REFERENCES stations(name),
                        station_geo TEXT,
                        operative BOOLEAN
                    )
                ''')
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")
            raise
        finally:
            self._release_connection(conn)
    
    def store_station_info(self, stations_df):
        """Store or update station information only if there are changes"""
        conn = self._get_connection()
        try:
            stations_df['last_updated'] = datetime.now()
            
            if self.db_type == 'postgres':
                cursor = conn.cursor()
                # Use ON CONFLICT for upsert (insert or update)
                data = [tuple(x) for x in stations_df[['name', 'lat', 'lon', 'capacity', 'last_updated']].to_numpy()]
                try:
                    cursor.executemany("""
                        INSERT INTO stations (name, lat, lon, capacity, last_updated)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (name) DO UPDATE
                        SET lat = EXCLUDED.lat,
                            lon = EXCLUDED.lon,
                            capacity = EXCLUDED.capacity,
                            last_updated = EXCLUDED.last_updated
                    """, data)
                    logger.info(f"Upserted {len(data)} stations (inserted or updated as needed)")
                except Exception as e:
                    logger.error(f"Error upserting stations: {str(e)}")
                    conn.rollback()
                    raise
                
                # Verify final station count
                cursor.execute("SELECT COUNT(*) FROM stations")
                final_count = cursor.fetchone()[0]
                logger.info(f"Total stations in database: {final_count}")
            else:
                # For SQLite, use pandas to_sql with if_exists='replace'
                stations_df.to_sql('stations', conn, if_exists='replace', index=False)
            
            conn.commit()
            logger.info(f"Successfully processed station information")
            
        except Exception as e:
            logger.error(f"Error storing station info: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            try:
                conn.rollback()
            except Exception:
                pass
        finally:
            self._release_connection(conn)
    
    def store_hourly_observations(self, observations_df):
        """Store hourly observations for all stations"""
        conn = self._get_connection()
        try:
            if self.db_type == 'postgres':
                # Convert DataFrame to list of tuples for PostgreSQL
                data = [tuple(x) for x in observations_df[['datetime', 'capacity', 'available_mechanical', 
                                                         'available_electrical', 'station_name', 
                                                         'station_geo', 'operative']].to_numpy()]
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT INTO hourly_observations 
                    (datetime, capacity, available_mechanical, available_electrical, 
                     station_name, station_geo, operative)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, data)
            else:
                observations_df.to_sql('hourly_observations', conn, if_exists='append', index=False)
            
            conn.commit()
            logger.info(f"Stored hourly observations for {len(observations_df)} stations")
        except Exception as e:
            logger.error(f"Error storing hourly observations: {str(e)}")
            raise
        finally:
            self._release_connection(conn)
    
    def get_last_24h_data(self, station_id):
        """Retrieve last 24 hours of data for a specific station"""
        conn = self._get_connection()
        try:
            query = """
            SELECT h.*, s.name, s.lat, s.lon, s.capacity
            FROM hourly_observations h
            JOIN stations s ON h.station_id = s.station_id
            WHERE h.station_id = %s
            AND h.timestamp >= NOW() - INTERVAL '24 hours'
            ORDER BY h.timestamp ASC
            """ if self.db_type == 'postgres' else """
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
        finally:
            self._release_connection(conn)
    
    def get_all_stations_last_24h(self):
        """Retrieve last 24 hours of data for all stations"""
        conn = self._get_connection()
        try:
            query = """
            SELECT h.*, s.name, s.lat, s.lon, s.capacity
            FROM hourly_observations h
            JOIN stations s ON h.station_id = s.station_id
            WHERE h.timestamp >= NOW() - INTERVAL '24 hours'
            ORDER BY h.station_id, h.timestamp ASC
            """ if self.db_type == 'postgres' else """
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
        finally:
            self._release_connection(conn)
    
    def cleanup_old_data(self, days_to_keep=30):
        """Remove data older than specified days"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if self.db_type == 'postgres':
                cursor.execute("""
                    DELETE FROM hourly_observations 
                    WHERE datetime < NOW() - INTERVAL '%s days'
                """, (days_to_keep,))
            else:
                cursor.execute("""
                    DELETE FROM hourly_observations 
                    WHERE datetime < datetime('now', '-? days')
                """, (days_to_keep,))
            
            conn.commit()
            logger.info(f"Cleaned up data older than {days_to_keep} days")
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
            raise
        finally:
            self._release_connection(conn)
    
    def __del__(self):
        """Cleanup connection pool when object is destroyed"""
        if hasattr(self, 'pool'):
            self.pool.closeall() 