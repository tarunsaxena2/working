"""
api.py — Prediction API
Contextual Predictive Maintenance — IoT Edge AI

Usage: uvicorn api:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import load_model, predict_single

app = FastAPI(title="Predictive Maintenance API", version="1.0")

# Load model once at startup (not on every request)
model = load_model()

# ---- Request Schema ----
class SensorReading(BaseModel):
    Air_temperature_K: float
    Process_temperature_K: float
    Rotational_speed_rpm: float
    Torque_Nm: float
    Tool_wear_min: float
    Type_enc: int
    ambient_temp_C: float
    factory_load_pct: float
    humidity_pct: float

# ---- Response Schema ----
class PredictionResponse(BaseModel):
    prediction: int
    probability: float

# ---- /predict endpoint ----
@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    input_dict = reading.dict()
    result = predict_single(input_dict, model)
    return result

@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    # Step 1: Convert request to dict
    input_dict = reading.dict()
    
    # Step 2: (Preprocessing hook - future me yahan validation/scaling add kar sakte hain)
    
    # Step 3: Run prediction
    result = predict_single(input_dict, model)
    
    # Step 4: Return structured response
    return result

@app.get("/")
def root():
    return {"message": "Predictive Maintenance API is running", "docs": "/docs"}