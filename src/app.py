from fastapi import FastAPI
import threading
from data.data_collector import VelibDataCollector
import logging
import time
import os
import uvicorn

app = FastAPI(title="Velib Data Collector API")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_collector():
    collector = VelibDataCollector()
    collector.collect_data()  # Collect initial data
    while True:
        collector.collect_data()
        time.sleep(3600)  # Sleep for 1 hour

@app.get("/")
async def home():
    return {"status": "Velib Data Collector is running!"}

@app.get("/health")
async def health():
    return {"status": "OK"}

if __name__ == '__main__':
    # Start the collector in a background thread
    collector_thread = threading.Thread(target=run_collector, daemon=True)
    collector_thread.start()
    
    # Start the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get('PORT', 8080))) 