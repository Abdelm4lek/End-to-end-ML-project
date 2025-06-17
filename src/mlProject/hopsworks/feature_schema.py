from typing import List, Dict
from mlProject.hopsworks.feature_store import HopsworksFeatureStore

VELIB_FEATURE_SCHEMA = {
    "datetime": "timestamp",
    "capacity": "int",
    "available_mechanical": "int",
    "available_electrical": "int",
    "station_name": "string",
    "station_geo": "string",
    "operative": "boolean"
}

def get_feature_group_schema() -> Dict:
    """Get the schema configuration for the Velib feature group."""
    return {
        "name": "velib_stations_status",
        "version": 1,
        "description": "Velib station status and information",
        "primary_key": ["station_name"],
        "online_enabled": True,
        "event_time": "datetime",
        "features": VELIB_FEATURE_SCHEMA
    } 