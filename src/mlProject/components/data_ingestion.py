import os
import urllib.request as request
import pandas as pd
from datetime import datetime, timedelta
from mlProject import logger
from mlProject.utils.common import get_size
from pathlib import Path
from mlProject.entity.config_entity import (DataIngestionConfig)


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def download_file(self):
        """Download data based on configured mode (local or production)"""
        if self.config.mode == "production":
            logger.info("Using production mode: Hopsworks Feature Store")
            self._fetch_from_hopsworks()
        else:
            logger.info("Using local mode: URL download")
            self._download_from_url()

    def _download_from_url(self):
        """Original URL download logic"""
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"RAR file already exists of size: {get_size(Path(self.config.local_data_file))}")

    def _fetch_from_hopsworks(self):
        """Fetch fresh data from Hopsworks Feature Store"""
        try:
            from mlProject.hopsworks.config import HopsworksConfig
            from mlProject.hopsworks.feature_store import HopsworksFeatureStore
            from mlProject.hopsworks.feature_schema import get_feature_group_schema
            
            logger.info("Connecting to Hopsworks Feature Store")
            hopsworks_config = HopsworksConfig()
            feature_store = HopsworksFeatureStore(hopsworks_config)
            
            # Calculate time window
            end_time = datetime.now()
            start_time = end_time - timedelta(days=self.config.training_window_days)
            
            logger.info(f"Fetching data from {start_time} to {end_time}")
            
            # Get feature group
            schema = get_feature_group_schema()
            feature_group = feature_store.get_feature_group(
                name=schema["name"],
                version=schema["version"]
            )
            
            # Query for fresh data
            query = feature_group.select_all().filter(
                feature_group.datetime >= start_time
            ).filter(
                feature_group.datetime <= end_time
            )
            
            fresh_data = query.read()
            
            if fresh_data.empty or len(fresh_data) < self.config.min_data_points:
                logger.warning(f"Not enough fresh data: {len(fresh_data)} records")
                raise ValueError(f"Insufficient data for training: {len(fresh_data)} < {self.config.min_data_points}")
            
            # Save to the expected CSV location
            fresh_data.to_csv(self.config.unzip_data_path, index=False)
            logger.info(f"Saved {len(fresh_data)} records to {self.config.unzip_data_path}")
            
        except Exception as e:
            logger.error(f"Failed to fetch from Hopsworks: {str(e)}")
            raise e

    def extract_archive_file(self):
        """
        Extracts the archive file (zip or rar) into the data directory
        Function returns None
        """
        if self.config.mode == "production":
            logger.info("Skipping extraction for production mode (data already saved as CSV)")
            return
            
        unzip_dir = self.config.unzip_dir
        os.makedirs(unzip_dir, exist_ok=True)
        
        # Check if the target directory already has the extracted file
        if os.path.exists(self.config.unzip_data_path):
            logger.info(f"Extracted file already exists in {unzip_dir}, skipping extraction")

        else:
            try:
                import patoolib
                patoolib.extract_archive(self.config.local_data_file, outdir=unzip_dir)
                logger.info(f"Successfully extracted archive to {unzip_dir}")
            except ImportError:
                raise ImportError("patoolib is required for RAR extraction in local mode. Install with: pip install patool")
            except Exception as e:
                logger.error(f"Error extracting archive: {str(e)}")
                raise e