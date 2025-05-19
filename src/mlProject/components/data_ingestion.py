import os
import urllib.request as request
import patoolib
from mlProject import logger
from mlProject.utils.common import get_size
from pathlib import Path
from mlProject.entity.config_entity import (DataIngestionConfig)


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"RAR file already exists of size: {get_size(Path(self.config.local_data_file))}")



    def extract_archive_file(self):
        """
        Extracts the archive file (zip or rar) into the data directory
        Function returns None
        """
        unzip_dir = self.config.unzip_dir
        os.makedirs(unzip_dir, exist_ok=True)
        
        # Check if the target directory already has the extracted file
        if os.path.exists(self.config.unzip_data_path):
            logger.info(f"Extracted file already exists in {unzip_dir}, skipping extraction")

        else:
            try:
                patoolib.extract_archive(self.config.local_data_file, outdir=unzip_dir)
                logger.info(f"Successfully extracted archive to {unzip_dir}")
            except Exception as e:
                logger.error(f"Error extracting archive: {str(e)}")
                raise e
  