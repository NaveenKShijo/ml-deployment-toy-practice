from fastapi import FastAPI, Request, HTTPException
import joblib
import numpy as np

app = FastAPI()

# SageMaker automatically extracts the model.tar.gz contents into /opt/ml/model
try:
    model = joblib.load("/opt/ml/model/model.pkl")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    model = None

@app.get("/ping")
def ping():
    # SageMaker requires a /ping endpoint for health checks
    if model is not None:
        return {"status": "Healthy"}
    return {"status": "Unhealthy", "reason": "Model not loaded"}

@app.post("/invocations")    
async def predict(request: Request):
    # if model is None:
    #     raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    # SageMaker calls the /invocations endpoint for predictions
    data = await request.json()
    
    # Assuming data is passed in a format that the model can understand
    if "features" in data:
        features = data["features"]
    else:
        features = data
        
    result = model.predict(features)
    
    # Convert numpy arrays to lists for JSON serialization
    if isinstance(result, np.ndarray):
        result = result.tolist()
        
    return {"result": result}
