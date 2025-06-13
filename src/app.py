from fastapi import FastAPI
import threading
from mlProject.hopsworks.velib_collector import VelibHopsworksCollector
import logging
import time
import os
import uvicorn
import signal
import sys
from datetime import datetime
import atexit

app = FastAPI(title="Velib Data Collector API")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for state management
running = True
last_collection_time = None
collector_thread = None
restart_count = 0
MAX_RESTARTS = 3

def run_collector():
    global last_collection_time, restart_count
    collector = VelibHopsworksCollector()
    
    while running:
        try:
            logger.info("Starting data collection cycle")
            collector.collect_data()
            last_collection_time = datetime.now()
            logger.info(f"Data collection completed at {last_collection_time}")
            restart_count = 0  # Reset restart count on successful collection
            time.sleep(3600)  # Sleep for 1 hour
        except Exception as e:
            logger.error(f"Error in collector thread: {e}")
            restart_count += 1
            if restart_count >= MAX_RESTARTS:
                logger.error("Maximum restart attempts reached. Shutting down.")
                os._exit(1)  # Force exit to trigger platform restart
            time.sleep(60)  # Wait a minute before retrying

@app.on_event("startup")
async def startup_event():
    global collector_thread
    collector_thread = threading.Thread(target=run_collector)
    collector_thread.start()

@app.on_event("shutdown")
async def shutdown_event():
    global running
    running = False
    if collector_thread:
        collector_thread.join()

@app.get("/")
async def home():
    return {
        "status": "Velib Data Collector is running!",
        "last_collection": str(last_collection_time) if last_collection_time else "No data collected yet",
        "restart_count": restart_count
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 