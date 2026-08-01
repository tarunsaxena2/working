"""
test_api.py — API Edge-Case Tests
Contextual Predictive Maintenance — IoT Edge AI

Usage: python -m pytest tests/test_api.py -v
(Make sure api.py is running: uvicorn api:app --reload)
"""

import requests

API_URL = "http://127.0.0.1:8000/predict"

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


def test_valid_payload_returns_200():
    response = requests.post(API_URL, json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "probability" in body


def test_missing_field_returns_422():
    payload = VALID_PAYLOAD.copy()
    del payload["Torque_Nm"]
    response = requests.post(API_URL, json=payload)
    assert response.status_code == 422


def test_negative_rpm_returns_422():
    payload = VALID_PAYLOAD.copy()
    payload["Rotational_speed_rpm"] = -50
    response = requests.post(API_URL, json=payload)
    assert response.status_code == 422


def test_negative_air_temp_returns_422():
    payload = VALID_PAYLOAD.copy()
    payload["Air_temperature_K"] = -10
    response = requests.post(API_URL, json=payload)
    assert response.status_code == 422


def test_out_of_range_type_enc_returns_422():
    payload = VALID_PAYLOAD.copy()
    payload["Type_enc"] = 5  # only 0, 1, 2 allowed
    response = requests.post(API_URL, json=payload)
    assert response.status_code == 422


def test_out_of_range_humidity_returns_422():
    payload = VALID_PAYLOAD.copy()
    payload["humidity_pct"] = 150  # max is 100
    response = requests.post(API_URL, json=payload)
    assert response.status_code == 422


def test_wrong_type_string_instead_of_float_returns_422():
    payload = VALID_PAYLOAD.copy()
    payload["Torque_Nm"] = "not_a_number"
    response = requests.post(API_URL, json=payload)
    assert response.status_code == 422


def test_empty_payload_returns_422():
    response = requests.post(API_URL, json={})
    assert response.status_code == 422
