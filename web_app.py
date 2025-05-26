import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime, timezone
import plotly.express as px
import plotly.graph_objects as go
from src.data.database import VelibDatabase
import joblib
import numpy as np
import pytz

# Set page configuration
st.set_page_config(
    page_title="Velib Demand Predictor",
    page_icon="🚲",
    layout="wide"
)

# Title and description
st.title("🚲 Velib Station Demand Predictor")
st.markdown("""
This application provides real-time predictions for Velib station demand using live data from the Velib API.
""")

# Initialize database connection
db = VelibDatabase(db_type='postgres')

# Load the trained model
@st.cache_resource
def load_model():
    try:
        return joblib.load('artifacts/model_trainer/model.joblib')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to prepare features for prediction
def prepare_features(station_data):
    """Prepare features for the model using last 24 hours of data"""
    # Sort by timestamp
    station_data = station_data.sort_values('timestamp')
    
    # Create lagged features (last 24 hours)
    features = []
    for i in range(1, 25):
        station_data[f'bikes_lag_{i}'] = station_data['num_bikes_available'].shift(i)
    
    # Drop rows with NaN values (first 24 hours)
    features_df = station_data.dropna()
    
    # Select only the feature columns
    feature_columns = [f'bikes_lag_{i}' for i in range(1, 25)]
    return features_df[feature_columns]

# Function to make predictions
def make_predictions(model, station_data):
    """Make predictions for all stations"""
    predictions = {}
    for station_id in station_data['station_id'].unique():
        station_df = station_data[station_data['station_id'] == station_id]
        if len(station_df) >= 24:  # Ensure we have enough data
            features = prepare_features(station_df)
            if not features.empty:
                pred = model.predict(features.iloc[-1:])[0]
                predictions[station_id] = pred
    return predictions

# Function to fetch station information (static data)
def fetch_station_info():
    try:
        url = "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_information.json"
        response = requests.get(url)
        data = response.json()
        return pd.DataFrame(data['data']['stations'])
    except Exception as e:
        st.error(f"Error fetching station information: {str(e)}")
        return None

# Function to fetch real-time Velib data
def fetch_velib_data():
    try:
        url = "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_status.json"
        response = requests.get(url)
        data = response.json()
        last_updated = datetime.fromtimestamp(data['data']['stations'][0]['last_reported'], tz=timezone.utc)
        return pd.DataFrame(data['data']['stations']), last_updated
    except Exception as e:
        st.error(f"Error fetching Velib data: {str(e)}")
        return None, None

# Function to process and display station data
def display_station_data(status_df, info_df, last_updated):
    if status_df is None or info_df is None:
        return
    
    # Display last update time in Paris timezone
    if last_updated:
        paris_tz = pytz.timezone('Europe/Paris')
        paris_time = last_updated.astimezone(paris_tz)
        st.caption(f"Last updated: {paris_time.strftime('%Y-%m-%d %H:%M:%S')} Paris time")
    
    # Merge the two dataframes on station_id
    merged_df = pd.merge(
        status_df,
        info_df[['station_id', 'name', 'lat', 'lon', 'capacity']],
        on='station_id',
        how='left'
    )
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Stations", len(merged_df))
    with col2:
        st.metric("Active Stations", merged_df['is_renting'].sum())
    with col3:
        st.metric("Total Bikes Available", merged_df['num_bikes_available'].sum())

    # Create a map visualization
    st.subheader("Station Locations and Current Availability")
    
    # Create custom hover template
    hover_template = """
    Station: <b>%{customdata[0]}</b><br>
    Coordinates: (%{lat:.4f}, %{lon:.4f})<br>
    Available Bikes: %{customdata[1]}<br>
    Available Docks: %{customdata[2]}<br>
    Total Capacity: %{customdata[3]}<br>
    <extra></extra>
    """
    
    fig = px.scatter_mapbox(
        merged_df,
        lat='lat',
        lon='lon',
        color='num_bikes_available',
        size='num_bikes_available',
        custom_data=['name', 'num_bikes_available', 'num_docks_available', 'capacity'],
        zoom=10,
        mapbox_style="carto-positron"
    )
    
    # Update hover template
    fig.update_traces(hovertemplate=hover_template)
    
    st.plotly_chart(fig, use_container_width=True)

    # Display station details in a table
    st.subheader("Station Details")
    st.dataframe(
        merged_df[['name', 'num_bikes_available', 'num_docks_available', 'is_renting', 'capacity']],
        use_container_width=True
    )

# Main app logic
def main():
    # Add a refresh button
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
    
    # Fetch both static and real-time data
    status_df, last_updated = fetch_velib_data()
    info_df = fetch_station_info()
    
    # Display the real-time data
    display_station_data(status_df, info_df, last_updated)

    # Add prediction section
    st.subheader("Demand Predictions")
    st.markdown("""
    ### Next Hour Predictions
    The predictions below show the expected demand for the next hour at each station.
    """)
    
    # Load model and make predictions
    model = load_model()
    
    if model is not None:
        # Get historical data for predictions
        current_data = db.get_all_stations_last_24h()
        
        if current_data is not None:
            # Make predictions
            predictions = make_predictions(model, current_data)
            
            # Get latest observations
            latest_data = current_data.groupby('station_id').last().reset_index()
            
            # Add predictions to the dataframe
            latest_data['predicted_bikes'] = latest_data['station_id'].map(predictions)
            
            # Create prediction map
            st.subheader("Predicted Station Demand")
            
            # Create custom hover template for predictions
            pred_hover_template = """
            Station: <b>%{customdata[0]}</b><br>
            Coordinates: (%{lat:.4f}, %{lon:.4f})<br>
            Current Bikes: %{customdata[1]}<br>
            Predicted Bikes: %{customdata[2]}<br>
            Available Docks: %{customdata[3]}<br>
            Total Capacity: %{customdata[4]}<br>
            <extra></extra>
            """
            
            pred_fig = px.scatter_mapbox(
                latest_data,
                lat='lat',
                lon='lon',
                color='predicted_bikes',
                size='predicted_bikes',
                custom_data=['name', 'num_bikes_available', 'predicted_bikes', 'num_docks_available', 'capacity'],
                zoom=10,
                mapbox_style="carto-positron"
            )
            
            # Update hover template
            pred_fig.update_traces(hovertemplate=pred_hover_template)
            
            st.plotly_chart(pred_fig, use_container_width=True)

            # Display predictions in a table
            st.subheader("Station Predictions")
            st.dataframe(
                latest_data[['name', 'num_bikes_available', 'predicted_bikes', 'num_docks_available', 'capacity']],
                use_container_width=True
            )
        else:
            st.error("No historical data available for predictions. Please ensure the data collector is running.")
    else:
        st.error("Model not loaded. Please ensure the model file exists and is valid.")

if __name__ == "__main__":
    main()
