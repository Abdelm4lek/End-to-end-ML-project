import hopsworks
import pandas as pd
from typing import Optional, List
from .config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    FEATURE_GROUP_NAME,
    FEATURE_VIEW_NAME
)



class HopsworksFeatureStore:
    def __init__(self):
        """Initialize connection to Hopsworks feature store."""
        self.project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT_NAME
        )
        self.fs = self.project.get_feature_store()
            
    def create_feature_group(self, name: str, version: int, description: str, 
                           primary_key: List[str], online_enabled: bool = True,
                           event_time: str = "datetime"):
        """Create a new feature group in Hopsworks."""
        return self.fs.create_feature_group(
            name=name,
            version=version,
            description=description,
            primary_key=primary_key,
            online_enabled=online_enabled,
            event_time=event_time
        )
    
    def append_data(self, feature_group_name: str, df: pd.DataFrame, version: int = 1):
        """Append data to an existing feature group."""
        feature_group = self.fs.get_feature_group(feature_group_name, version)
        feature_group.insert(df)
        return feature_group
    
    def get_feature_view(self) -> Optional[hopsworks.feature_view.FeatureView]:
        """Get the feature view for velib data prediction."""
        try:
            return self.fs.get_feature_view(
                name=FEATURE_VIEW_NAME,
                version=1
            )
        except:
            return None
    
    def create_feature_view(self, feature_group: hopsworks.feature_group.FeatureGroup):
        """Create a feature view from the feature group."""
        feature_view = self.fs.create_feature_view(
            name=FEATURE_VIEW_NAME,
            version=1,
            query=feature_group.select_all()
        )
        return feature_view
    
    def get_training_data(self, start_time: Optional[str] = None, end_time: Optional[str] = None) -> pd.DataFrame:
        """Get training data from the feature view."""
        feature_view = self.get_feature_view()
        if feature_view is None:
            raise ValueError("Feature view not found. Please create it first.")
        
        return feature_view.get_batch_data(
            start_time=start_time,
            end_time=end_time
        )
    
    def get_feature_vector(self, feature_vector: List[float]) -> pd.DataFrame:
        """Get feature vector for online prediction."""
        feature_view = self.get_feature_view()
        if feature_view is None:
            raise ValueError("Feature view not found. Please create it first.")
        
        return feature_view.get_feature_vector(feature_vector) 