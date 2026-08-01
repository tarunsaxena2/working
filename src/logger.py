"""
logger.py — Prediction Logging
Contextual Predictive Maintenance — IoT Edge AI

Logs every prediction (timestamp, sensor inputs, prediction, probability)
to a CSV file for later review in the dashboard's log viewer.

Usage:
    from src.logger import log_prediction
"""

import os
import csv
from datetime import datetime

LOG_PATH = 'logs/predictions_log.csv'

LOG_FIELDS = [
    'timestamp',
    'Air_temperature_K',
    'Process_temperature_K',
    'Rotational_speed_rpm',
    'Torque_Nm',
    'Tool_wear_min',
    'Type_enc',
    'ambient_temp_C',
    'factory_load_pct',
    'humidity_pct',
    'prediction',
    'probability',
]

def _ensure_log_file():
    """Create the logs/ folder and CSV file with headers if they don't exist yet."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            writer.writeheader()

def log_prediction(input_dict: dict, prediction: int, probability: float):
    """
    Append one prediction record to the CSV log.

    Parameters:
        input_dict (dict): the 9 sensor/context features sent to the model
        prediction (int): 0 or 1
        probability (float): model's failure probability
    """
    _ensure_log_file()

    row = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        **input_dict,
        'prediction': prediction,
        'probability': probability,
    }

    with open(LOG_PATH, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writerow(row)

if __name__ == "__main__":
    # Quick sanity test
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
    log_prediction(sample, prediction=0, probability=0.0015)
    print(f"✅ Logged test prediction to {LOG_PATH}")
