from fastapi import FastAPI, Request, HTTPException
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# SageMaker automatically extracts the model.tar.gz contents into /opt/ml/model
try:
    model = joblib.load("/opt/ml/model/model.pkl")
    scaler = joblib.load("/opt/ml/model/scaler.pkl")
except Exception as e:
    print(f"Warning: Could not load model or scaler: {e}")
    model = None
    scaler = None

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
