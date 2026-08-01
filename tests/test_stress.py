"""
test_stress.py — API Stress Test
Contextual Predictive Maintenance — IoT Edge AI

Sends rapid-fire valid + malformed requests to confirm the API
stays stable under load and doesn't crash.

Usage: python tests/test_stress.py
(Make sure api.py is running: uvicorn api:app --reload)
"""

import requests
import time
import random

API_URL = "http://127.0.0.1:8000/predict"
HEALTH_URL = "http://127.0.0.1:8000/health"

VALID_PAYLOAD = {
    "Air_temperature_K": 298.5,
    "Process_temperature_K": 308.7,
    "Rotational_speed_rpm": 1500,
    "Torque_Nm": 40.2,
    "Tool_wear_min": 10,
    "Type_enc": 1,
    "ambient_temp_C": 28.0,
    "factory_load_pct": 75.0,
    "humidity_pct": 60.0
}

MALFORMED_PAYLOADS = [
    {},  # empty
    {"Torque_Nm": "not_a_number"},  # wrong type, missing fields
    {**VALID_PAYLOAD, "Rotational_speed_rpm": -999},  # negative
    {**VALID_PAYLOAD, "Type_enc": 99},  # out of range
    {**VALID_PAYLOAD, "humidity_pct": None},  # null value
    "not_even_json",  # completely wrong shape (will fail to send as json properly, handled separately)
]

def rapid_fire_valid(n=50):
    print(f"\n=== Rapid-fire {n} valid requests ===")
    success, fail = 0, 0
    start = time.time()
    for i in range(n):
        try:
            r = requests.post(API_URL, json=VALID_PAYLOAD, timeout=5)
            if r.status_code == 200:
                success += 1
            else:
                fail += 1
                print(f"  Unexpected status {r.status_code} on request {i+1}")
        except Exception as e:
            fail += 1
            print(f"  Request {i+1} raised exception: {e}")
    elapsed = time.time() - start
    print(f"Success: {success}/{n} | Failed: {fail}/{n} | Total time: {elapsed:.2f}s")

def malformed_bombardment(rounds=20):
    print(f"\n=== Sending {rounds} rounds of malformed requests ===")
    handled, crashed = 0, 0
    for i in range(rounds):
        payload = random.choice(MALFORMED_PAYLOADS[:-1])  # skip the non-json one for requests.post(json=)
        try:
            r = requests.post(API_URL, json=payload, timeout=5)
            if r.status_code in (422, 500):
                handled += 1
            else:
                print(f"  Round {i+1}: unexpected status {r.status_code}")
        except Exception as e:
            crashed += 1
            print(f"  Round {i+1} raised exception: {e}")
    print(f"Handled gracefully: {handled}/{rounds} | Connection errors: {crashed}/{rounds}")

def check_health_after_load():
    print(f"\n=== Checking /health after load ===")
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        print(f"Health check: {r.status_code} — {r.json()}")
        assert r.status_code == 200
        print("✅ API survived the stress test and is still healthy.")
    except Exception as e:
        print(f"❌ API did not respond to health check: {e}")

if __name__ == "__main__":
    rapid_fire_valid(n=50)
    malformed_bombardment(rounds=20)
    check_health_after_load()
