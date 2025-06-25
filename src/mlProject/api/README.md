# Velib Hourly Prediction API

A simplified FastAPI service for hourly batch Velib station demand predictions.

## Overview

This API provides **hourly batch predictions** for all Velib stations in Paris using your existing ML infrastructure:
- **Reuses your existing `Predictor` class** from `src/mlProject/components/prediction.py`
- **Loads models** from Hopsworks Model Registry (with local fallback)
- **Fetches data** using your existing Hopsworks Feature Store integration
- **Simple focus**: Only batch predictions for all stations every hour

## API Endpoints

### Core Endpoint
- `POST /predict/all` - Get hourly predictions for **all** Velib stations

### System Endpoints
- `GET /health` - Health check (model loaded, Hopsworks connection)
- `POST /model/reload` - Reload model from registry
- `GET /` - API information

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables (Optional)
```bash
export HOPSWORKS_API_KEY=your_api_key
export HOPSWORKS_PROJECT_NAME=your_project_name
export HOPSWORKS_HOST=your_hopsworks_host
```
*Note: If not set, API will use local model fallback*

### 3. Run the API
```bash
# Simple development mode
python src/mlProject/api/run.py

# Or with uvicorn
uvicorn src.mlProject.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access API
- **Swagger UI**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Example Usage

### Get All Station Predictions
```bash
curl -X POST "http://localhost:8000/predict/all" \
     -H "Content-Type: application/json"
```

**Response:**
```json
{
  "predictions": {
    "Benjamin Godard - Victor Hugo": 12.0,
    "Dupleix - Grenelle": 8.0,
    "Bir-Hakeim - Rapp": 15.0,
    ...
  },
  "total_stations": 1400,
  "successful_predictions": 1350,
  "timestamp": "2024-01-15T14:30:00",
  "status": "success",
  "model_info": {
    "model_loaded": true,
    "model_version": "1",
    "loaded_at": "2024-01-15T14:00:00",
    "model_type": "LGBMBooster"
  }
}
```

## Architecture

### Simplified Design
```
FastAPI Main App (main.py)
    ↓
PredictionService (prediction_service.py)
    ↓
Your Existing Predictor Class (components/prediction.py)
    ↓
Your Existing Hopsworks Integration
```

### Key Components

1. **`prediction_service.py`**
   - Thin wrapper around your existing `Predictor` class
   - Handles Hopsworks model loading with local fallback
   - Simple async interface

2. **`main.py`**
   - Minimal FastAPI app with 4 endpoints
   - Focus on batch prediction use case
   - Clean error handling and logging

3. **Reused Components**
   - `Predictor` class for all prediction logic
   - `HopsworksModelRegistry` for model loading
   - `HopsworksFeatureStore` for data access

## Production Deployment

Perfect for **hourly prediction services** on:
- **Railway** (recommended)
- **Render** 
- **Google Cloud Run**

### Environment Variables for Production
```bash
HOPSWORKS_API_KEY=your_production_api_key
HOPSWORKS_PROJECT_NAME=your_project_name
HOPSWORKS_HOST=your_hopsworks_host
```

## Scheduling Hourly Predictions

### Option 1: External Scheduler (Recommended)
```bash
# Cron job to call API every hour
0 * * * * curl -X POST "https://your-api.railway.app/predict/all"
```

### Option 2: GitHub Actions
```yaml
# .github/workflows/hourly_predictions.yml
on:
  schedule:
    - cron: '0 * * * *'  # Every hour
jobs:
  predict:
    runs-on: ubuntu-latest
    steps:
      - name: Call prediction API
        run: curl -X POST "${{ secrets.API_URL }}/predict/all"
```

## Benefits of This Approach

✅ **Reuses your existing code** - No duplication of prediction logic
✅ **Simple and focused** - Only what you actually need
✅ **Production ready** - Built on your proven components  
✅ **Fallback support** - Works without Hopsworks if needed
✅ **Easy to scale** - Single endpoint, clear responsibility

## Monitoring

The API includes health monitoring:
- Model loading status
- Hopsworks connectivity 
- Prediction success rates
- Detailed error logging 