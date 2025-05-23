import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

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
        return pd.DataFrame(data['data']['stations'])
    except Exception as e:
        st.error(f"Error fetching Velib data: {str(e)}")
        return None

# Function to process and display station data
def display_station_data(status_df, info_df):
    if status_df is None or info_df is None:
        return
    
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
    st.subheader("Station Locations and Availability")
    
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
    status_df = fetch_velib_data()
    info_df = fetch_station_info()
    
    # Display the combined data
    display_station_data(status_df, info_df)

    # Add prediction section
    st.subheader("Demand Predictions")
    st.markdown("""
    ### Next Hour Predictions
    The predictions below show the expected demand for the next hour at each station.
    """)
    
    # Placeholder for predictions (to be implemented with your ML model)
    st.info("ML model predictions will be integrated here")

if __name__ == "__main__":
    main()
