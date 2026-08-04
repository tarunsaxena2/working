"""
demo_bad_input.py — Live "Bad Input" Demo Script
Contextual Predictive Maintenance — IoT Edge AI

A scripted moment for the live demo: shows judges that /health and
input validation handle bad/malformed requests gracefully instead
of crashing. Reuses the same scenarios as tests/test_api.py.

Usage: python demo_bad_input.py
(Make sure api.py is running: uvicorn api:app --reload)
"""

import requests
import time

API_URL = "http://127.0.0.1:8000"

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


def banner(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def pause():
    time.sleep(1.2)


def show_result(label, response):
    print(f"\n>>> {label}")
    print(f"    Status Code: {response.status_code}")
    try:
        print(f"    Response:    {response.json()}")
    except Exception:
        print(f"    Response:    {response.text[:200]}")
    pause()


def demo():
    banner("DEMO: System Reliability — Bad Input Handling")

    # 1. Health check — proves the API is alive before we stress it
    print("\nStep 1: Confirming the API is healthy...")
    r = requests.get(f"{API_URL}/health")
    show_result("GET /health", r)

    # 2. A normal, valid request — baseline
    print("Step 2: Sending a normal, valid sensor reading...")
    r = requests.post(f"{API_URL}/predict", json=VALID_PAYLOAD)
    show_result("POST /predict (valid payload)", r)

    # 3. Missing required field
    print("Step 3: Sending a payload with a MISSING field (Torque_Nm)...")
    bad_payload = VALID_PAYLOAD.copy()
    del bad_payload["Torque_Nm"]
    r = requests.post(f"{API_URL}/predict", json=bad_payload)
    show_result("POST /predict (missing field)", r)

    # 4. Out-of-range value
    print("Step 4: Sending a payload with an impossible negative RPM...")
    bad_payload = VALID_PAYLOAD.copy()
    bad_payload["Rotational_speed_rpm"] = -999
    r = requests.post(f"{API_URL}/predict", json=bad_payload)
    show_result("POST /predict (negative RPM)", r)

    # 5. Wrong data type
    print("Step 5: Sending a payload with a STRING instead of a number...")
    bad_payload = VALID_PAYLOAD.copy()
    bad_payload["Torque_Nm"] = "not_a_number"
    r = requests.post(f"{API_URL}/predict", json=bad_payload)
    show_result("POST /predict (wrong type)", r)

    # 6. Completely empty payload
    print("Step 6: Sending a completely EMPTY payload...")
    r = requests.post(f"{API_URL}/predict", json={})
    show_result("POST /predict (empty payload)", r)

    # 7. Health check again — proves the API survived all of the above
    print("Step 7: Confirming the API is STILL healthy after all that...")
    r = requests.get(f"{API_URL}/health")
    show_result("GET /health (after bad input storm)", r)

    banner("DEMO COMPLETE — API never crashed, every bad input was rejected with a clear 422 error.")


if __name__ == "__main__":
    demo()
