from fastapi import FastAPI
import threading
from data.data_collector import VelibDataCollector
import logging
import time
import os
import uvicorn
import signal
import sys
from datetime import datetime

app = FastAPI(title="Velib Data Collector API")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for state management
running = True
last_collection_time = None
collector_thread = None

def run_collector():
    global last_collection_time
    collector = VelibDataCollector()
    
    while running:
        try:
            logger.info("Starting data collection cycle")
            collector.collect_data()
            last_collection_time = datetime.now()
            logger.info(f"Data collection completed at {last_collection_time}")
            time.sleep(3600)  # Sleep for 1 hour
        except Exception as e:
            logger.error(f"Error in collector thread: {e}")
            time.sleep(60)  # Wait a minute before retrying

def signal_handler(signum, frame):
    global running
    logger.info("Received shutdown signal")
    running = False
    # Give the collector thread time to finish current operation
    time.sleep(5)
    sys.exit(0)

@app.get("/")
async def home():
    return {
        "status": "Velib Data Collector is running!",
        "last_collection": str(last_collection_time) if last_collection_time else "No data collected yet"
    }

@app.get("/health")
async def health():
    return {
        "status": "OK",
        "collector_running": collector_thread and collector_thread.is_alive(),
        "last_collection": str(last_collection_time) if last_collection_time else "No data collected yet"
    }

def start_collector():
    global collector_thread
    collector_thread = threading.Thread(target=run_collector, daemon=True)
    collector_thread.start()
    logger.info("Collector thread started")

if __name__ == '__main__':
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start the collector
    start_collector()
    
    # Start the FastAPI app
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting FastAPI server on port {port}")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port, 
        log_level="info",
        access_log=True
    ) 