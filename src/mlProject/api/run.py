"""
Script to run the FastAPI application.
"""

import uvicorn
import os
import sys
from pathlib import Path

# Add project root to Python path
src_root = Path(__file__).resolve().parents[2]  # /app/src
sys.path.insert(0, str(src_root))

def run_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = True):
    """
    Run the FastAPI application.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
    """
    uvicorn.run(
        "src.mlProject.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    # Set environment variables for development if not set
    if not os.getenv("HOPSWORKS_API_KEY"):
        print("Warning: HOPSWORKS_API_KEY not set. API will use fallback local model.")
    
    print("Starting Velib Prediction API...")
    print("API Documentation available at: http://localhost:8000/docs")
    print("Health check available at: http://localhost:8000/health")
    
    run_api() 