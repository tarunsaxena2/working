# Predictive Maintenance API — Documentation

Base URL (local): `http://127.0.0.1:8000`

Interactive Swagger UI available at: `/docs`

---

## 1. `GET /`

**Description:** Root endpoint — confirms the API is running.

**Request:** No parameters required.

**Example Request:**
```bash
curl http://127.0.0.1:8000/
```

**Example Response (200 OK):**
```json
{
  "message": "Predictive Maintenance API is running",
  "docs": "/docs"
}
```

---

## 2. `GET /health`

**Description:** Health check endpoint — confirms the API and model are ready to serve requests.

**Request:** No parameters required.

**Example Request:**
```bash
curl http://127.0.0.1:8000/health
```

**Example Response (200 OK):**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## 3. `POST /predict`

**Description:** Takes a single sensor reading and returns the failure prediction and probability.

**Request Body (JSON):**

| Field | Type | Constraints | Description |
|---|---|---|---|
| `Air_temperature_K` | float | > 0 | Air temperature in Kelvin |
| `Process_temperature_K` | float | > 0 | Process temperature in Kelvin |
| `Rotational_speed_rpm` | float | > 0 | Rotational speed in RPM |
| `Torque_Nm` | float | ≥ 0 | Torque in Newton-metres |
| `Tool_wear_min` | float | ≥ 0 | Tool wear in minutes |
| `Type_enc` | int | 0, 1, or 2 | Encoded machine type (H=0, L=1, M=2) |
| `ambient_temp_C` | float | — | Ambient temperature in Celsius |
| `factory_load_pct` | float | 0–100 | Factory load percentage |
| `humidity_pct` | float | 0–100 | Humidity percentage |

**Example Request:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Air_temperature_K": 298.5,
    "Process_temperature_K": 308.7,
    "Rotational_speed_rpm": 1500,
    "Torque_Nm": 40.2,
    "Tool_wear_min": 10,
    "Type_enc": 1,
    "ambient_temp_C": 28.0,
    "factory_load_pct": 75.0,
    "humidity_pct": 60.0
  }'
```

**Example Response (200 OK):**
```json
{
  "prediction": 0,
  "probability": 0.0015050230507364201
}
```

- `prediction`: `0` = Healthy, `1` = Failure predicted
- `probability`: Model's confidence that a failure will occur (0.0–1.0)

**Error Response — Validation Error (422):**

Returned when a field is missing or out of the allowed range (e.g. negative RPM):
```json
{
  "detail": [
    {
      "type": "greater_than",
      "loc": ["body", "Rotational_speed_rpm"],
      "msg": "Input should be greater than 0",
      "input": -50
    }
  ]
}
```

**Error Response — Server Error (500):**

Returned if prediction fails internally (e.g. model error):
```json
{
  "detail": "Prediction failed: <error message>"
}
```

---

## Notes for Integration

- Field names are case-sensitive and must exactly match the table above (no trailing underscores).
- `Type_enc` must be a pre-encoded integer (0/1/2), not a raw string like "L"/"M"/"H". Use `sensor_mapping.py`'s `SensorMapper` to convert raw sensor data (including machine type) before calling this API.
- The model is loaded once at API startup — restart the server if `models/lgbm_retrained.pkl` is updated.
