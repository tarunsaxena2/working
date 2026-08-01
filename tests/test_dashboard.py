"""
test_dashboard.py — Dashboard & Stream Stress Tests
Vaibhav Gautam — Dashboard & Integration
Week 4 Day 3 Task

Tests simulate_stream.py behavior under:
- Dropped connections
- Bad sensor readings
- Out-of-range values
"""

import requests
import time

API_URL = "http://127.0.0.1:8000"

def test_bad_sensor_readings():
    """Test API handles out-of-range sensor values gracefully."""
    print("=== Testing Bad Sensor Readings ===")

    bad_payloads = [
        # Missing field
        {
            "Air_temperature_K": 300.0,
            "Process_temperature_K": 310.0,
        },
        # Negative values
        {
            "Air_temperature_K": -999.0,
            "Process_temperature_K": -999.0,
            "Rotational_speed_rpm": -1.0,
            "Torque_Nm": -1.0,
            "Tool_wear_min": -1.0,
            "Type_enc": 0,
            "ambient_temp_C": 28.0,
            "factory_load_pct": 75.0,
            "humidity_pct": 60.0,
        },
        # Wrong data types
        {
            "Air_temperature_K": "not_a_number",
            "Process_temperature_K": 310.0,
            "Rotational_speed_rpm": 1500.0,
            "Torque_Nm": 40.0,
            "Tool_wear_min": 100.0,
            "Type_enc": 0,
            "ambient_temp_C": 28.0,
            "factory_load_pct": 75.0,
            "humidity_pct": 60.0,
        },
    ]

    for i, payload in enumerate(bad_payloads):
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=3)
            print(f"Bad payload {i+1}: Status {resp.status_code} — {'✅ Handled' if resp.status_code in [200, 422] else '❌ Unexpected'}")
        except requests.exceptions.ConnectionError:
            print(f"Bad payload {i+1}: ❌ API not running")
            break
        except Exception as e:
            print(f"Bad payload {i+1}: ❌ Error: {e}")

def test_dropped_connection():
    """Test behavior when API is unavailable."""
    print("\n=== Testing Dropped Connection ===")
    try:
        resp = requests.get("http://127.0.0.1:9999/health", timeout=2)
        print("Wrong port: Unexpected response")
    except requests.exceptions.ConnectionError:
        print("Wrong port: ✅ Connection refused — handled gracefully")
    except requests.exceptions.Timeout:
        print("Wrong port: ✅ Timeout — handled gracefully")

def test_rapid_fire():
    """Send rapid requests to test API stability."""
    print("\n=== Testing Rapid-Fire Requests ===")
    payload = {
        "Air_temperature_K": 300.0,
        "Process_temperature_K": 310.0,
        "Rotational_speed_rpm": 1500.0,
        "Torque_Nm": 40.0,
        "Tool_wear_min": 100.0,
        "Type_enc": 1,
        "ambient_temp_C": 28.0,
        "factory_load_pct": 75.0,
        "humidity_pct": 60.0,
    }
    success = 0
    for i in range(20):
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=3)
            if resp.status_code == 200:
                success += 1
        except Exception:
            pass
    print(f"Rapid-fire: {success}/20 requests succeeded ✅")

if __name__ == "__main__":
    print("Starting dashboard stress tests...\n")
    test_dropped_connection()
    test_bad_sensor_readings()
    test_rapid_fire()
    print("\n✅ All stress tests complete!")