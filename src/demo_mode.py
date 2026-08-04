"""
demo_mode.py — Offline / Demo Mode Fallback
Vaibhav Gautam — Dashboard & Integration
Week 3 Day 2 Task

Provides cached fallback predictions when the live API
is not reachable — so the demo never fails on stage.

Usage:
    from src.demo_mode import get_demo_prediction, is_api_reachable
"""

import numpy as np
import pandas as pd

API_URL = "http://127.0.0.1:8000"

# =================================================================
# CACHED DEMO PREDICTIONS
# Pre-computed from the real model so numbers are always correct
# =================================================================
DEMO_PREDICTIONS = [
    {"probability": 0.0002, "prediction": 0, "status": "healthy"},
    {"probability": 0.0015, "prediction": 0, "status": "healthy"},
    {"probability": 0.0340, "prediction": 0, "status": "healthy"},
    {"probability": 0.4820, "prediction": 0, "status": "elevated"},
    {"probability": 0.7650, "prediction": 1, "status": "critical"},
    {"probability": 0.0008, "prediction": 0, "status": "healthy"},
    {"probability": 0.3100, "prediction": 0, "status": "elevated"},
    {"probability": 0.9200, "prediction": 1, "status": "critical"},
]

# Cached fleet predictions for demo mode
DEMO_FLEET = [
    {"machine_name": "CNC Mill A",  "probability": 0.0520, "prediction": 0},
    {"machine_name": "Lathe B",     "probability": 0.7800, "prediction": 1},
    {"machine_name": "Press C",     "probability": 0.3400, "prediction": 0},
    {"machine_name": "Grinder D",   "probability": 0.0085, "prediction": 0},
]


def is_api_reachable(timeout: float = 2.0) -> bool:
    """
    Check if the live API is reachable.

    Parameters:
        timeout (float): Connection timeout in seconds

    Returns:
        bool: True if API is reachable, False otherwise
    """
    try:
        import requests
        resp = requests.get(f"{API_URL}/health", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def get_demo_prediction(index: int = 0) -> dict:
    """
    Get a cached demo prediction when API is offline.

    Parameters:
        index (int): Which demo prediction to return

    Returns:
        dict: prediction and probability
    """
    idx = index % len(DEMO_PREDICTIONS)
    return DEMO_PREDICTIONS[idx]


def get_demo_fleet() -> list:
    """
    Get cached fleet predictions for demo mode.

    Returns:
        list: List of machine predictions
    """
    return DEMO_FLEET


def predict_with_fallback(payload: dict, index: int = 0) -> tuple:
    """
    Try live API first, fall back to demo mode if unreachable.

    Parameters:
        payload (dict): Sensor reading payload
        index (int): Demo prediction index for fallback

    Returns:
        tuple: (result dict, is_live bool)
    """
    try:
        import requests
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=3)
        if resp.status_code == 200:
            return resp.json(), True
    except Exception:
        pass

    # Fallback to demo mode
    return get_demo_prediction(index), False


if __name__ == "__main__":
    print("=== Demo Mode Test ===")
    print(f"API reachable: {is_api_reachable()}")
    print(f"Demo prediction: {get_demo_prediction(0)}")
    print(f"Demo fleet: {get_demo_fleet()}")
    print("✅ Demo mode working!")