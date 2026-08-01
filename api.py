"""
api.py — Prediction API
Contextual Predictive Maintenance — IoT Edge AI

Usage: uvicorn api:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.predict import load_model, predict_single
from src.logger import log_prediction

app = FastAPI(title="Predictive Maintenance API", version="1.0")

# Load model once at startup (not on every request)
model = load_model()

# Warm-up: run one dummy prediction to avoid cold-start latency on first real request
_warmup_sample = {
    "Air_temperature_K": 300.0, "Process_temperature_K": 310.0,
    "Rotational_speed_rpm": 1500.0, "Torque_Nm": 40.0,
    "Tool_wear_min": 0.0, "Type_enc": 1,
    "ambient_temp_C": 28.0, "factory_load_pct": 75.0, "humidity_pct": 60.0
}
predict_single(_warmup_sample, model)

# ---- Request Schema ----
class SensorReading(BaseModel):
    Air_temperature_K: float = Field(..., gt=0, description="Air temperature in Kelvin")
    Process_temperature_K: float = Field(..., gt=0, description="Process temperature in Kelvin")
    Rotational_speed_rpm: float = Field(..., gt=0, description="Rotational speed in RPM")
    Torque_Nm: float = Field(..., ge=0, description="Torque in Nm")
    Tool_wear_min: float = Field(..., ge=0, description="Tool wear in minutes")
    Type_enc: int = Field(..., ge=0, le=2, description="Encoded machine type (0, 1, or 2)")
    ambient_temp_C: float
    factory_load_pct: float = Field(..., ge=0, le=100, description="Factory load percentage")
    humidity_pct: float = Field(..., ge=0, le=100, description="Humidity percentage")

# ---- Response Schema ----
class PredictionResponse(BaseModel):
    prediction: int
    probability: float

# ---- Root endpoint ----
@app.get("/")
def root():
    return {"message": "Predictive Maintenance API is running", "docs": "/docs"}

# ---- Health check endpoint ----
@app.get("/health")
def health_check():
    """Check if API and model are ready to serve requests."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

# ---- /predict endpoint ----
@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    try:
        input_dict = reading.dict()
        result = predict_single(input_dict, model)
        log_prediction(input_dict, result['prediction'], result['probability'])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
