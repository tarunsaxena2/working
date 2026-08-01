"""
test_latency.py — Measure /predict endpoint latency
"""
import requests
import time

API_URL = "http://127.0.0.1:8000/predict"

sample = {
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

session = requests.Session()

times = []
n_requests = 20

for i in range(n_requests):
    start = time.time()
    response = session.post(API_URL, json=sample, timeout=5)
    elapsed = (time.time() - start) * 1000  # ms
    times.append(elapsed)
    print(f"Request {i+1}: {elapsed:.2f} ms (status: {response.status_code})")

print(f"\n=== Latency Summary ===")
print(f"Average: {sum(times)/len(times):.2f} ms")
print(f"Min: {min(times):.2f} ms")
print(f"Max: {max(times):.2f} ms")
