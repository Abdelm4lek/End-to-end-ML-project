import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('DB_credentials.env')

# Hopsworks configuration
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST")

# Feature Store configuration
FEATURE_GROUP_NAME = "Velib_data_features"
FEATURE_VIEW_NAME = "Velib_data_feature_view"

# Model Registry configuration
MODEL_NAME = "Velib_demand_model"
MODEL_VERSION = 1.0