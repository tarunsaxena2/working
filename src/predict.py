"""
predict.py — Prediction Helper Functions
Contextual Predictive Maintenance — IoT Edge AI

Usage:
    from src.predict import predict_single, predict_batch
"""

import pandas as pd
import joblib

MODEL_PATH = 'models/lgbm_retrained.pkl'

def load_model(filepath=MODEL_PATH):
    """Load the trained pipeline from disk."""
    return joblib.load(filepath)

def predict_single(input_dict, model=None):
    """
    Predict failure probability for a single sensor reading.

    Parameters:
        input_dict (dict): e.g. {
            'Air_temperature_K': 298.5,
            'Process_temperature_K': 308.7,
            'Rotational_speed_rpm': 1500,
            'Torque_Nm': 40.2,
            'Tool_wear_min': 10,
            'Type_enc': 1,
            'ambient_temp_C': 28.0,
            'factory_load_pct': 75.0,
            'humidity_pct': 60.0
        }
        model: preloaded pipeline (optional, loads fresh if None)

    Returns:
        dict: {'prediction': int, 'probability': float}
    """
    if model is None:
        model = load_model()

    X = pd.DataFrame([input_dict])

    # Single inference call — derive prediction from probability
    # instead of calling predict() and predict_proba() separately.
    prob = model.predict_proba(X)[0][1]
    pred = int(prob >= 0.5)

    return {
        'prediction': pred,
        'probability': float(prob)
    }

def predict_batch(input_df, model=None):
    """
    Predict failure for a batch of sensor readings.

    Parameters:
        input_df (pd.DataFrame): multiple rows, same columns as training features
        model: preloaded pipeline (optional, loads fresh if None)

    Returns:
        pd.DataFrame: original rows + 'prediction' + 'probability' columns
    """
    if model is None:
        model = load_model()

    probs = model.predict_proba(input_df)[:, 1]
    preds = (probs >= 0.5).astype(int)

    result = input_df.copy()
    result['prediction'] = preds
    result['probability'] = probs

    return result

if __name__ == "__main__":
    # quick sanity test
    model = load_model()
    sample = {
        'Air_temperature_K': 298.5,
        'Process_temperature_K': 308.7,
        'Rotational_speed_rpm': 1500,
        'Torque_Nm': 40.2,
        'Tool_wear_min': 10,
        'Type_enc': 1,
        'ambient_temp_C': 28.0,
        'factory_load_pct': 75.0,
        'humidity_pct': 60.0
    }
    print("Single prediction test:", predict_single(sample, model))
