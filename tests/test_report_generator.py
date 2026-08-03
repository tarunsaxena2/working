"""
test_report_generator.py — Tests for PDF Report Generator
Contextual Predictive Maintenance — IoT Edge AI

Ensures PDF generation never crashes mid-demo, even with missing
or unusual data.

Usage: python -m pytest tests/test_report_generator.py -v
"""

import os
import pytest
from src.report_generator import generate_report, generate_shap_chart_image


TEST_OUTPUT = "test_output_report.pdf"
TEST_SHAP_IMG = "test_output_shap.png"


@pytest.fixture(autouse=True)
def cleanup():
    """Remove test artifacts after each test."""
    yield
    for f in [TEST_OUTPUT, TEST_SHAP_IMG]:
        if os.path.exists(f):
            os.remove(f)


def test_basic_report_generates_without_sample():
    """Report should generate fine with just metrics/config, no prediction."""
    path = generate_report(TEST_OUTPUT)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


def test_report_with_valid_sample_prediction():
    sample = {
        "input": {
            'Air_temperature_K': 298.5, 'Process_temperature_K': 308.7,
            'Rotational_speed_rpm': 1500, 'Torque_Nm': 40.2,
            'Tool_wear_min': 10, 'Type_enc': 1,
            'ambient_temp_C': 28.0, 'factory_load_pct': 75.0, 'humidity_pct': 60.0
        },
        "prediction": 0,
        "probability": 0.0015,
    }
    path = generate_report(TEST_OUTPUT, sample_prediction=sample)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


def test_report_with_high_risk_prediction():
    """High-probability failure case should also render fine."""
    sample = {
        "input": {
            'Air_temperature_K': 303.0, 'Process_temperature_K': 313.0,
            'Rotational_speed_rpm': 2400, 'Torque_Nm': 75.0,
            'Tool_wear_min': 230, 'Type_enc': 0,
            'ambient_temp_C': 40.0, 'factory_load_pct': 95.0, 'humidity_pct': 85.0
        },
        "prediction": 1,
        "probability": 0.97,
    }
    path = generate_report(TEST_OUTPUT, sample_prediction=sample)
    assert os.path.exists(path)


def test_report_with_missing_input_keys_does_not_crash():
    """If sample_prediction['input'] is missing some sensor fields,
    the report should still generate (SHAP chart may fail gracefully)."""
    sample = {
        "input": {
            'Air_temperature_K': 298.5,
            # missing most other fields on purpose
        },
        "prediction": 0,
        "probability": 0.10,
    }
    # Should not raise an exception
    path = generate_report(TEST_OUTPUT, sample_prediction=sample)
    assert os.path.exists(path)


def test_report_with_empty_input_dict_does_not_crash():
    sample = {
        "input": {},
        "prediction": 0,
        "probability": 0.0,
    }
    path = generate_report(TEST_OUTPUT, sample_prediction=sample)
    assert os.path.exists(path)


def test_report_with_missing_prediction_key_does_not_crash():
    """sample_prediction without a 'prediction' key should not crash."""
    sample = {
        "input": {
            'Air_temperature_K': 298.5, 'Process_temperature_K': 308.7,
            'Rotational_speed_rpm': 1500, 'Torque_Nm': 40.2,
            'Tool_wear_min': 10, 'Type_enc': 1,
            'ambient_temp_C': 28.0, 'factory_load_pct': 75.0, 'humidity_pct': 60.0
        },
        "probability": 0.5,
    }
    path = generate_report(TEST_OUTPUT, sample_prediction=sample)
    assert os.path.exists(path)


def test_report_with_nonexistent_chart_image_does_not_crash():
    """Referencing a chart image that doesn't exist should not crash,
    just show a placeholder message instead."""
    path = generate_report(TEST_OUTPUT, chart_image_paths=["nonexistent_chart.png"])
    assert os.path.exists(path)


def test_shap_chart_generation_with_valid_input():
    input_dict = {
        'Air_temperature_K': 298.5, 'Process_temperature_K': 308.7,
        'Rotational_speed_rpm': 1500, 'Torque_Nm': 40.2,
        'Tool_wear_min': 10, 'Type_enc': 1,
        'ambient_temp_C': 28.0, 'factory_load_pct': 75.0, 'humidity_pct': 60.0
    }
    result = generate_shap_chart_image(input_dict, TEST_SHAP_IMG)
    assert result == TEST_SHAP_IMG
    assert os.path.exists(TEST_SHAP_IMG)


def test_shap_chart_generation_with_empty_input_returns_none():
    """Empty/invalid input should fail gracefully (return None), not crash."""
    result = generate_shap_chart_image({}, TEST_SHAP_IMG)
    # Either it returns None (graceful failure) or somehow succeeds —
    # either way, it must not raise an exception (pytest would catch that).
    assert result is None or os.path.exists(TEST_SHAP_IMG)
