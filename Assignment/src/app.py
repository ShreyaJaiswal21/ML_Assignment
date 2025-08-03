# src/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
import os

# --- Model Loading ---
# It's better to load the model once at startup.
# Make sure to set the MLFLOW_TRACKING_URI if your runs are not local.
# For this project, we assume the 'mlruns' directory is present.
try:
    # Find the latest run in the default experiment (experiment_id='0')
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=["0"], order_by=["start_time DESC"], max_results=1)
    if not runs:
        raise FileNotFoundError("No MLflow runs found.")
    
    latest_run_id = runs[0].info.run_id
    logged_model_uri = f'runs:/{latest_run_id}/price_prediction_model'
    print(f"Loading model from: {logged_model_uri}")
    loaded_model = mlflow.pyfunc.load_model(logged_model_uri)
except Exception as e:
    print(f"Error loading model: {e}. The API will not work.")
    loaded_model = None

# --- API Application ---
app = FastAPI(title="Real Estate Price Prediction API")

# Define the input data schema for robust validation
class PropertyFeatures(BaseModel):
    propertyType: str
    city: str
    visitOrCount: int

@app.get("/")
def read_root():
    return {"message": "Welcome to the Price Prediction API. Use the /predict endpoint to make predictions."}

@app.post("/predict")
def predict(features: PropertyFeatures):
    """
    Accepts property features in a JSON payload and returns the predicted price.
    """
    if not loaded_model:
        raise HTTPException(status_code=503, detail="Model is not available. Please check server logs.")
        
    try:
        # Convert the input Pydantic model to a DataFrame
        input_df = pd.DataFrame([features.dict()])
        
        # Get the prediction
        prediction = loaded_model.predict(input_df)
        
        # Return the prediction in a JSON response
        return {"predicted_price": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")

