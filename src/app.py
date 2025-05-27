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
import atexit

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
restart_count = 0
MAX_RESTARTS = 3

def run_collector():
    global last_collection_time, restart_count
    collector = VelibDataCollector()
    
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

def cleanup():
    global running
    logger.info("Performing cleanup before shutdown")
    running = False
    if collector_thread and collector_thread.is_alive():
        collector_thread.join(timeout=5)
    logger.info("Cleanup completed")

def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}")
    cleanup()
    sys.exit(0)

@app.get("/")
async def home():
    return {
        "status": "Velib Data Collector is running!",
        "last_collection": str(last_collection_time) if last_collection_time else "No data collected yet",
        "restart_count": restart_count
    }

@app.get("/health")
async def health():
    is_healthy = (
        collector_thread is not None 
        and collector_thread.is_alive() 
        and last_collection_time is not None 
        and (datetime.now() - last_collection_time).total_seconds() < 7200  # 2 hours
    )
    return {
        "status": "OK" if is_healthy else "DEGRADED",
        "collector_running": collector_thread and collector_thread.is_alive(),
        "last_collection": str(last_collection_time) if last_collection_time else "No data collected yet",
        "restart_count": restart_count
    }

def start_collector():
    global collector_thread
    collector_thread = threading.Thread(target=run_collector, daemon=True)
    collector_thread.start()
    logger.info("Collector thread started")

if __name__ == '__main__':
    # Register signal handlers and cleanup
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(cleanup)
    
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
        access_log=True,
        workers=1  # Ensure single worker to prevent multiple collectors
    ) 