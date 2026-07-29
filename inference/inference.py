from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
import joblib
import pandas as pd
import numpy as np

import traceback
import os
import time

# Global references - loaded at startup
model = None
scaler = None

MODEL_DIR = "/opt/ml/model"


def load_model_artifacts():
    """Attempt to load model artifacts with retries, since SageMaker
    may still be extracting model.tar.gz when the container starts."""
    global model, scaler

    print(f"=== Model Loading Debug Info ===")
    print(f"MODEL_DIR: {MODEL_DIR}")
    print(f"Current working directory: {os.getcwd()}")

    # List common paths for debugging
    for check_path in [MODEL_DIR, "/opt/ml", "/opt/ml/code", "/opt/ml/code/models"]:
        if os.path.exists(check_path):
            try:
                contents = os.listdir(check_path)
                print(f"  {check_path} exists, contents: {contents}")
            except Exception as e:
                print(f"  {check_path} exists but cannot list: {e}")
        else:
            print(f"  {check_path} does NOT exist")

    # Retry loading in case SageMaker hasn't finished extracting yet
    max_retries = 5
    for attempt in range(max_retries):
        try:
            model_path = os.path.join(MODEL_DIR, "model.pkl")
            scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

            if not os.path.exists(model_path):
                print(f"  Attempt {attempt+1}: {model_path} not found yet")
                time.sleep(2)
                continue

            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            print(f"Model and scaler loaded successfully on attempt {attempt+1}.")
            return True
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            traceback.print_exc()
            time.sleep(2)

    print(f"ERROR: Failed to load model after {max_retries} attempts.")

    # Last resort: try loading from /opt/ml/code/models (where COPY . . puts them)
    fallback_dir = "/opt/ml/code/models"
    print(f"Trying fallback path: {fallback_dir}")
    try:
        if os.path.exists(fallback_dir):
            model = joblib.load(os.path.join(fallback_dir, "model.pkl"))
            scaler = joblib.load(os.path.join(fallback_dir, "scaler.pkl"))
            print(f"Model and scaler loaded from FALLBACK path: {fallback_dir}")
            return True
    except Exception as e:
        print(f"Fallback load also failed: {e}")
        traceback.print_exc()

    return False


@asynccontextmanager
async def lifespan(app):
    """Load model when FastAPI starts up."""
    load_model_artifacts()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/ping")
def ping():
    # SageMaker requires a /ping endpoint for health checks
    if model is not None and scaler is not None:
        return {"status": "Healthy"}
    return {"status": "Unhealthy", "reason": "Model or scaler not loaded"}

@app.post("/invocations")    
async def predict(request: Request):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler is not loaded.")
        
    # SageMaker calls the /invocations endpoint for predictions
    data = await request.json()
    
    # Assuming data is passed in a format that the model can understand
    if "features" in data:
        raw_features = data["features"]
    else:
        raw_features = data
        
    # Convert input to DataFrame if it's a dict or list
    if isinstance(raw_features, dict):
        df_features = pd.DataFrame([raw_features])
    elif isinstance(raw_features, list):
        df_features = pd.DataFrame(raw_features)
    else:
        df_features = raw_features

    # Scale the features using the loaded ColumnTransformer scaler
    scaled_features = scaler.transform(df_features)
        
    result = model.predict(scaled_features)
    
    # Convert numpy arrays to lists for JSON serialization
    if isinstance(result, np.ndarray):
        result = result.tolist()
        
    return {"result": result}

