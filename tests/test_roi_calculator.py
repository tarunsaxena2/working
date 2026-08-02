"""
test_roi_calculator.py — Unit Tests for ROI Calculator
Contextual Predictive Maintenance — IoT Edge AI

Usage: python -m pytest tests/test_roi_calculator.py -v
"""

import pytest
from src.roi_calculator import estimate_savings


def test_normal_case():
    result = estimate_savings(
        downtime_cost_per_hour=5000,
        hours_downtime_avoided=8,
        failure_probability=0.87
    )
    assert result['estimated_savings'] == 34800.0


def test_zero_risk_returns_zero_savings():
    result = estimate_savings(
        downtime_cost_per_hour=5000,
        hours_downtime_avoided=8,
        failure_probability=0.0
    )
    assert result['estimated_savings'] == 0.0


def test_full_risk_returns_full_product():
    result = estimate_savings(
        downtime_cost_per_hour=1000,
        hours_downtime_avoided=10,
        failure_probability=1.0
    )
    assert result['estimated_savings'] == 10000.0


def test_zero_downtime_cost_returns_zero():
    result = estimate_savings(
        downtime_cost_per_hour=0,
        hours_downtime_avoided=10,
        failure_probability=0.9
    )
    assert result['estimated_savings'] == 0.0


def test_zero_hours_avoided_returns_zero():
    result = estimate_savings(
        downtime_cost_per_hour=5000,
        hours_downtime_avoided=0,
        failure_probability=0.9
    )
    assert result['estimated_savings'] == 0.0


def test_negative_downtime_cost_raises_error():
    with pytest.raises(ValueError):
        estimate_savings(
            downtime_cost_per_hour=-100,
            hours_downtime_avoided=5,
            failure_probability=0.5
        )


def test_negative_hours_avoided_raises_error():
    with pytest.raises(ValueError):
        estimate_savings(
            downtime_cost_per_hour=5000,
            hours_downtime_avoided=-2,
            failure_probability=0.5
        )


def test_probability_above_one_raises_error():
    with pytest.raises(ValueError):
        estimate_savings(
            downtime_cost_per_hour=5000,
            hours_downtime_avoided=5,
            failure_probability=1.5
        )


def test_probability_below_zero_raises_error():
    with pytest.raises(ValueError):
        estimate_savings(
            downtime_cost_per_hour=5000,
            hours_downtime_avoided=5,
            failure_probability=-0.1
        )


def test_result_contains_all_expected_keys():
    result = estimate_savings(
        downtime_cost_per_hour=2000,
        hours_downtime_avoided=4,
        failure_probability=0.6
    )
    expected_keys = {
        'estimated_savings',
        'downtime_cost_per_hour',
        'hours_downtime_avoided',
        'failure_probability'
    }
    assert set(result.keys()) == expected_keys
