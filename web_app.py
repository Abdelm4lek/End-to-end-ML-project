import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timezone
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pytz
import time
import logging
from typing import Dict, Optional, List
import subprocess
import sys



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Velib Demand Predictor",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = "http://localhost:8000"
REFRESH_INTERVAL = st.sidebar.selectbox("Auto-refresh interval (seconds)", [30, 60, 120, 300], index=1)

def start_fastapi_if_needed():
    """Start FastAPI in the background if it's not already running"""
    try:
        # Check if FastAPI is already running
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if response.status_code == 200:
            return True  # Already running
    except:
        pass
    
    # FastAPI is not running, start it
    try:
        st.info("🚀 Starting API client service in background...")
        
        # Start FastAPI process
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "src.mlProject.api.main:app", 
            "--host", "127.0.0.1", 
            "--port", "8000"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait for it to start (up to 15 seconds)
        for i in range(15):
            try:
                response = requests.get(f"{API_BASE_URL}/health", timeout=2)
                if response.status_code == 200:
                    st.success("✅ API client started successfully!")
                    return True
            except:
                pass
            time.sleep(1)
        
        st.error("❌ Failed to start API client. Please start it manually.")
        return False
        
    except Exception as e:
        st.error(f"❌ Error starting API client: {str(e)}")
        return False

class VelibAPIClient:
    """Client for interacting with FastAPI and external Velib API"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 30
        
    def health_check(self) -> Dict:
        """Check API health"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"status": "unhealthy", "error": str(e)}
    
    def get_predictions(self) -> Dict:
        """Get predictions from FastAPI"""
        try:
            response = self.session.post(f"{self.base_url}/predict/all")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Prediction request failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def reload_model(self) -> Dict:
        """Reload model via FastAPI"""
        try:
            response = self.session.post(f"{self.base_url}/model/reload")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Model reload failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def fetch_station_info(_self):
        """Fetch station information (static data) - with fallback"""
        try:
            url = "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_information.json"
            response = _self.session.get(url)
            response.raise_for_status()
            data = response.json()
            return pd.DataFrame(data['data']['stations'])
        except Exception as e:
            logger.error(f"Error fetching station info: {str(e)}")
            st.error(f"Failed to fetch station information: {str(e)}")
            return pd.DataFrame()

    @st.cache_data(ttl=60)  # Cache for 1 minute
    def fetch_velib_data(_self):
        """Fetch real-time Velib data - with fallback"""
        try:
            url = "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_status.json"
            response = _self.session.get(url)
            response.raise_for_status()
            data = response.json()
            last_updated = datetime.fromtimestamp(data['data']['stations'][0]['last_reported'], tz=timezone.utc)
            return pd.DataFrame(data['data']['stations']), last_updated
        except Exception as e:
            logger.error(f"Error fetching Velib data: {str(e)}")
            st.error(f"Failed to fetch real-time data: {str(e)}")
            return pd.DataFrame(), None

@st.cache_resource
def get_api_client():
    """Get API client instance"""
    return VelibAPIClient(API_BASE_URL)

def display_api_status():
    """Display API status in sidebar"""
    api_client = get_api_client()
    
    st.sidebar.markdown("### 🔧 System Status")
    
    # Check FastAPI health
    health = api_client.health_check()
    
    if health.get("status") == "healthy":
        st.sidebar.success("✅ API client: Healthy")
        st.sidebar.info(f"🤖 Model Loaded: {'✅' if health.get('model_loaded') else '❌'}")
        st.sidebar.info(f"📊 Hopsworks: {'✅' if health.get('hopsworks_connected') else '❌'}")
    elif health.get("status") == "degraded":
        st.sidebar.warning("⚠️ API client: Degraded")
        st.sidebar.info(f"🤖 Model Loaded: {'✅' if health.get('model_loaded') else '❌'}")
        st.sidebar.info(f"📊 Hopsworks: {'✅' if health.get('hopsworks_connected') else '❌'}")
    else:
        st.sidebar.error("❌ API client: Unhealthy")
        st.sidebar.error(f"Error: {health.get('error', 'Unknown error')}")
        
        # Auto-start button
        if st.sidebar.button("🚀 Start API client"):
            start_fastapi_if_needed()
            st.rerun()
        
        # Model reload button if API is accessible
        if st.sidebar.button("🔄 Reload Model"):
            with st.spinner("Reloading model..."):
                result = api_client.reload_model()
                if "error" not in result:
                    st.sidebar.success("Model reloaded successfully!")
                    st.rerun()
                else:
                    st.sidebar.error(f"Failed to reload model: {result['error']}")

def create_performance_dashboard(predictions_data: Dict):
    """Create model performance dashboard"""
    st.subheader("📊 Model Performance Dashboard")
    
    if not predictions_data or "predictions" not in predictions_data:
        st.warning("No prediction data available for dashboard")
        return
    
    predictions = predictions_data["predictions"]
    
    # Model info
    model_info = predictions_data.get("model_info", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Stations", predictions_data.get("total_stations", 0))
    
    with col2:
        st.metric("Successful Predictions", predictions_data.get("successful_predictions", 0))
    
    with col3:
        success_rate = (predictions_data.get("successful_predictions", 0) / 
                       max(predictions_data.get("total_stations", 1), 1)) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col4:
        st.metric("Model Version", model_info.get("version", "Unknown"))
    
    # Prediction distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prediction Distribution")
        pred_values = [v for v in predictions.values() if v is not None]
        if pred_values:
            fig = px.histogram(x=pred_values, nbins=30, title="Distribution of Predicted Bike Counts")
            fig.update_layout(xaxis_title="Predicted Bikes", yaxis_title="Number of Stations")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No valid predictions to display")
    
    with col2:
        st.subheader("Prediction Statistics")
        if pred_values:
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                'Value': [
                    np.mean(pred_values),
                    np.median(pred_values),
                    np.std(pred_values),
                    np.min(pred_values),
                    np.max(pred_values)
                ]
            })
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.warning("No valid predictions to analyze")

def create_station_comparison_tool(merged_df: pd.DataFrame, predictions: Dict):
    """Create station comparison tool"""
    st.subheader("🔍 Station Comparison Tool")
    
    if merged_df.empty:
        st.warning("No station data available for comparison")
        return
    
    # Station selection
    stations = sorted(merged_df['name'].dropna().unique())
    selected_stations = st.multiselect(
        "Select stations to compare (max 5):",
        stations,
        default=stations[:3] if len(stations) >= 3 else stations,
        max_selections=5
    )
    
    if not selected_stations:
        st.info("Please select at least one station to compare")
        return
    
    # Filter data for selected stations
    comparison_df = merged_df[merged_df['name'].isin(selected_stations)].copy()
    
    # Add predictions
    comparison_df['predicted_bikes'] = comparison_df['name'].map(
        lambda x: predictions.get(x, None) if predictions else None
    )
    
    # Create comparison table
    cols_to_show = ['name', 'num_bikes_available', 'predicted_bikes', 'num_docks_available', 
                    'capacity', 'is_renting']
    
    st.subheader("Comparison Table")
    st.dataframe(comparison_df[cols_to_show], use_container_width=True)
    
    # Visual comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # Current vs Predicted bikes
        if 'predicted_bikes' in comparison_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Current Bikes',
                x=comparison_df['name'],
                y=comparison_df['num_bikes_available'],
                marker_color='lightblue'
            ))
            
            predicted_values = comparison_df['predicted_bikes'].fillna(0)
            fig.add_trace(go.Bar(
                name='Predicted Bikes',
                x=comparison_df['name'],
                y=predicted_values,
                marker_color='orange'
            ))
            
            fig.update_layout(
                title='Current vs Predicted Bikes',
                xaxis_title='Station',
                yaxis_title='Number of Bikes',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Capacity utilization
        comparison_df['utilization'] = (comparison_df['num_bikes_available'] / 
                                      comparison_df['capacity'] * 100)
        
        fig = px.bar(
            comparison_df,
            x='name',
            y='utilization',
            title='Current Capacity Utilization (%)',
            color='utilization',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(xaxis_title='Station', yaxis_title='Utilization (%)')
        st.plotly_chart(fig, use_container_width=True)

def create_historical_trends():
    """Create historical trends analysis (placeholder for future implementation)"""
    st.subheader("📈 Historical Trends Analysis")
    
    st.info("""
    **Coming Soon: Historical Trends**
    
    This section will include:
    - 📊 Station demand patterns over time
    - 🕐 Hourly/daily/weekly trends
    - 🌡️ Weather correlation analysis
    - 📅 Seasonal variations
    - 🎯 Prediction accuracy metrics over time
    
    *Integration with Hopsworks Feature Store for historical data is in progress.*
    """)
    
    # Sample chart to demonstrate future functionality
    sample_dates = pd.date_range(start='2024-01-01', end='2024-01-07', freq='H')
    sample_data = pd.DataFrame({
        'datetime': sample_dates,
        'demand': np.random.normal(15, 5, len(sample_dates)) + 
                 10 * np.sin(2 * np.pi * sample_dates.hour / 24)  # Daily pattern
    })
    
    fig = px.line(sample_data, x='datetime', y='demand', 
                  title='Sample: Station Demand Pattern (Demo Data)')
    fig.update_layout(xaxis_title='Time', yaxis_title='Bike Demand')
    st.plotly_chart(fig, use_container_width=True)

def create_prediction_map(merged_df: pd.DataFrame, predictions: Dict):
    """Create prediction map"""
    if merged_df.empty or not predictions:
        st.warning("No data available for prediction map")
        return
    
    # Add predictions to dataframe
    merged_df = merged_df.copy()
    merged_df['predicted_bikes'] = merged_df['name'].map(predictions)
    
    # Filter out stations without predictions
    prediction_df = merged_df.dropna(subset=['predicted_bikes'])
    
    if prediction_df.empty:
        st.warning("No stations have valid predictions")
        return
    
    # Create prediction map
    st.subheader("🗺️ Predicted Station Demand Map")
    
    # Prediction difference (predicted - current)
    prediction_df['prediction_diff'] = (prediction_df['predicted_bikes'] - 
                                      prediction_df['num_bikes_available'])
    
    # Custom hover template
    hover_template = """
    <b>%{customdata[0]}</b><br>
    📍 (%{lat:.4f}, %{lon:.4f})<br>
    🚲 Current: %{customdata[1]}<br>
    🔮 Predicted: %{customdata[2]}<br>
    📈 Change: %{customdata[3]:+.0f}<br>
    🚗 Available Docks: %{customdata[4]}<br>
    📊 Capacity: %{customdata[5]}<br>
    <extra></extra>
    """
    
    fig = px.scatter_mapbox(
        prediction_df,
        lat='lat',
        lon='lon',
        color='predicted_bikes',
        size='predicted_bikes',
        custom_data=['name', 'num_bikes_available', 'predicted_bikes', 
                    'prediction_diff', 'num_docks_available', 'capacity'],
        zoom=10,
        mapbox_style="carto-positron",
        title="Predicted Bike Availability",
        color_continuous_scale="Viridis"
    )
    
    fig.update_traces(hovertemplate=hover_template)
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction changes map
    st.subheader("📊 Predicted Changes Map")
    
    fig2 = px.scatter_mapbox(
        prediction_df,
        lat='lat',
        lon='lon',
        color='prediction_diff',
        size=abs(prediction_df['prediction_diff']).clip(lower=1),
        custom_data=['name', 'num_bikes_available', 'predicted_bikes', 
                    'prediction_diff', 'num_docks_available', 'capacity'],
        zoom=10,
        mapbox_style="carto-positron",
        title="Predicted Changes (+: increase, -: decrease)",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0
    )
    
    fig2.update_traces(hovertemplate=hover_template)
    st.plotly_chart(fig2, use_container_width=True)

def main():
    """Main application logic"""
    
    # Title and description
    st.title("🚲 Velib Demand Predictor")
    st.markdown("""
    **ML Predictions** dashboard for Velib bike-sharing demand.
    """)
    
    # Auto-start FastAPI if needed (only on first load)
    if 'fastapi_checked' not in st.session_state:
        st.session_state.fastapi_checked = True
        start_fastapi_if_needed()
    
    # Display API status in sidebar
    display_api_status()
    
    # Auto-refresh functionality
    if st.sidebar.checkbox("🔄 Auto-refresh", value=False):
        time.sleep(REFRESH_INTERVAL)
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()
    
    # Navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Real-time Overview", 
        "🔮 Predictions", 
        "📊 Model Performance", 
        "🔍 Station Comparison", 
        "📈 Historical Trends"
    ])
    
    # Get API client
    api_client = get_api_client()
    
    # Fetch data
    with st.spinner("Loading station data..."):
        status_df, last_updated = api_client.fetch_velib_data()
        info_df = api_client.fetch_station_info()
    
    # Check if we have data
    if status_df.empty or info_df.empty:
        st.error("⚠️ **Fallback Mode:** Unable to fetch real-time station data. Please check your connection.")
        st.stop()
    
    # Merge dataframes
    merged_df = pd.merge(
        status_df,
        info_df[['station_id', 'name', 'lat', 'lon', 'capacity']],
        on='station_id',
        how='left'
    )
    
    with tab1:
        # Real-time overview
        st.subheader("📍 Real-time Station Overview")
        
        # Display last update time
        if last_updated:
            paris_tz = pytz.timezone('Europe/Paris')
            paris_time = last_updated.astimezone(paris_tz)
            st.caption(f"🕐 Last updated: {paris_time.strftime('%Y-%m-%d %H:%M:%S')} Paris time")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Stations", len(merged_df))
        with col2:
            st.metric("Active Stations", merged_df['is_renting'].sum())
        with col3:
            st.metric("Total Bikes Available", merged_df['num_bikes_available'].sum())
        with col4:
            avg_utilization = (merged_df['num_bikes_available'] / merged_df['capacity'] * 100).mean()
            st.metric("Avg Utilization", f"{avg_utilization:.1f}%")

        # Current availability map
        st.subheader("🗺️ Current Station Availability")
        
        hover_template = """
        <b>%{customdata[0]}</b><br>
        📍 (%{lat:.4f}, %{lon:.4f})<br>
        🚲 Available: %{customdata[1]}<br>
        🚗 Docks: %{customdata[2]}<br>
        📊 Capacity: %{customdata[3]}<br>
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
            mapbox_style="carto-positron",
            color_continuous_scale="Viridis"
        )
        
        fig.update_traces(hovertemplate=hover_template)
        st.plotly_chart(fig, use_container_width=True)

        # Station details table
        st.subheader("📋 Station Details")
        display_df = merged_df[['name', 'num_bikes_available', 'num_docks_available', 
                               'is_renting', 'capacity']].copy()
        display_df['utilization_%'] = (display_df['num_bikes_available'] / 
                                     display_df['capacity'] * 100).round(1)
        st.dataframe(display_df, use_container_width=True)
    
    with tab2:
        # Predictions tab
        st.subheader("🔮 Next Hour Predictions")
        
        with st.spinner("Getting predictions from API client..."):
            predictions_data = api_client.get_predictions()
        
        if "error" in predictions_data:
            st.error(f"⚠️ **Prediction Error:** {predictions_data['error']}")
            st.info("💡 **Fallback:** You can still view real-time station data in the Overview tab.")
        else:
            predictions = predictions_data.get("predictions", {})
            
            if predictions:
                st.success(f"✅ Successfully generated predictions for {len(predictions)} stations")
                
                # Create prediction maps
                create_prediction_map(merged_df, predictions)
                
                # Prediction table
                st.subheader("📊 Detailed Predictions")
                pred_df = pd.DataFrame([
                    {
                        'Station': station,
                        'Current Bikes': merged_df[merged_df['name'] == station]['num_bikes_available'].iloc[0] 
                                       if not merged_df[merged_df['name'] == station].empty else 0,
                        'Predicted Bikes': int(prediction),
                        'Change': int(prediction) - (merged_df[merged_df['name'] == station]['num_bikes_available'].iloc[0] 
                                                   if not merged_df[merged_df['name'] == station].empty else 0)
                    }
                    for station, prediction in predictions.items()
                ])
                
                # Sort by change (biggest increases first)
                pred_df = pred_df.sort_values('Change', ascending=False)
                st.dataframe(pred_df, use_container_width=True)
            else:
                st.warning("No predictions available. Check model status in the sidebar.")
    
    with tab3:
        # Model performance dashboard
        if "predictions_data" in locals() and "error" not in predictions_data:
            create_performance_dashboard(predictions_data)
        else:
            st.warning("Model performance data not available. Make sure predictions are working.")
    
    with tab4:
        # Station comparison tool
        if "predictions_data" in locals() and "error" not in predictions_data:
            create_station_comparison_tool(merged_df, predictions_data.get("predictions", {}))
        else:
            create_station_comparison_tool(merged_df, {})
    
    with tab5:
        # Historical trends
        create_historical_trends()

if __name__ == "__main__":
    main()
