"""
simulate_stream.py — Streams CSV rows to /predict every 2-3 seconds
Vaibhav Gautam — Dashboard & Integration
Week 2 Day 3 Task

Usage: python simulate_stream.py
Make sure api.py is running first: python api.py
"""

import pandas as pd
import numpy as np
import requests
import time
import random
import re
from sklearn.preprocessing import LabelEncoder

# API endpoint
API_URL = "http://localhost:8000/predict"

# Load dataset
def load_stream_data():
    df = pd.read_csv("data/ai4i2020.csv")
    le = LabelEncoder()
    df["Type_enc"] = le.fit_transform(df["Type"])
    np.random.seed(42)
    df["ambient_temp_C"]   = np.random.normal(loc=28,  scale=5,  size=len(df))
    df["factory_load_pct"] = np.random.uniform(50, 100, size=len(df))
    df["humidity_pct"]     = np.random.normal(loc=60,  scale=10, size=len(df))
    return df

def clean_col(c):
    return re.sub(r"[^A-Za-z0-9_]+", "_", c)

def row_to_payload(row):
    """Convert a DataFrame row to API payload."""
    return {
        "Air_temperature_K_":    float(row["Air temperature [K]"]),
        "Process_temperature_K_": float(row["Process temperature [K]"]),
        "Rotational_speed_rpm_": float(row["Rotational speed [rpm]"]),
        "Torque_Nm_":            float(row["Torque [Nm]"]),
        "Tool_wear_min_":        float(row["Tool wear [min]"]),
        "Type_enc":              int(row["Type_enc"]),
        "ambient_temp_C":        float(row["ambient_temp_C"]),
        "factory_load_pct":      float(row["factory_load_pct"]),
        "humidity_pct":          float(row["humidity_pct"]),
    }

def stream_to_api():
    print("=== Sensor Stream Simulator Starting ===")
    print(f"Streaming to: {API_URL}")
    print("Press Ctrl+C to stop\n")

    df = load_stream_data()
    total_rows = len(df)
    sent = 0
    failures_caught = 0

    for idx, row in df.iterrows():
        try:
            payload = row_to_payload(row)
            response = requests.post(API_URL, json=payload, timeout=5)

            if response.status_code == 200:
                result = response.json()
                prob   = result.get("failure_probability", 0)
                pred   = result.get("prediction", 0)
                actual = int(row["Machine failure"])

                status = "🚨 FAILURE" if pred == 1 else "✅ HEALTHY"
                match  = "✓" if pred == actual else "✗"

                if pred == 1:
                    failures_caught += 1

                print(
                    f"Row {idx+1:05d}/{total_rows} | "
                    f"{status} | "
                    f"Prob: {prob:.4f} | "
                    f"Actual: {actual} | "
                    f"Match: {match}"
                )
            else:
                print(f"Row {idx+1} | ❌ API Error: {response.status_code} — {response.text}")

            sent += 1

            # Stream every 2-3 seconds
            delay = random.uniform(2, 3)
            time.sleep(delay)

        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to API at {API_URL}")
            print("Make sure api.py is running: python api.py")
            break
        except KeyboardInterrupt:
            print(f"\n\n=== Stream Stopped ===")
            print(f"Rows sent:       {sent}")
            print(f"Failures caught: {failures_caught}")
            break

    print(f"\n=== Stream Complete ===")
    print(f"Total rows sent: {sent}")
    print(f"Failures caught: {failures_caught}")

if __name__ == "__main__":
    stream_to_api()