"""
test_model_comparison.py — Tests for Model Comparison Data Loader
Contextual Predictive Maintenance — IoT Edge AI

Ensures the ablation study / model comparison data never silently
breaks or returns malformed data before demo day.

Usage: python -m pytest tests/test_model_comparison.py -v

Note: These tests actually train Random Forest models (via
get_model_comparison_data / get_feature_importance_comparison),
so they take longer than typical unit tests (~20-40s total).
"""

import pandas as pd
import pytest
from src.model_comparison_data import (
    get_model_comparison_data,
    get_feature_importance_comparison,
    PRODUCTION_MODEL_RESULTS,
)


@pytest.fixture(scope="module")
def comparison_df():
    """Run the comparison data loader once and reuse across tests (it's slow)."""
    return get_model_comparison_data()


def test_returns_dataframe(comparison_df):
    assert isinstance(comparison_df, pd.DataFrame)


def test_has_expected_columns(comparison_df):
    expected_cols = {"model", "feature_set", "macro_f1", "precision", "recall"}
    assert expected_cols.issubset(set(comparison_df.columns))


def test_no_feature_importances_leak_into_dataframe(comparison_df):
    """feature_importances is a Series and should never end up as a DataFrame column."""
    assert "feature_importances" not in comparison_df.columns


def test_has_three_rows(comparison_df):
    """Two Random Forest variants + one production LightGBM row."""
    assert len(comparison_df) == 3


def test_contains_production_model_row(comparison_df):
    prod_rows = comparison_df[comparison_df["model"] == "LightGBM + SMOTE (Production)"]
    assert len(prod_rows) == 1
    assert prod_rows.iloc[0]["macro_f1"] == PRODUCTION_MODEL_RESULTS["macro_f1"]


def test_all_scores_are_valid_probabilities(comparison_df):
    """Macro F1, precision, recall should all be between 0 and 1."""
    for col in ["macro_f1", "precision", "recall"]:
        assert comparison_df[col].between(0, 1).all(), f"{col} has values outside [0, 1]"


def test_no_null_values(comparison_df):
    assert not comparison_df.isnull().any().any()


def test_production_model_meets_kpi_target(comparison_df):
    """Sanity check: our production model should still meet the 0.85 target."""
    prod_f1 = comparison_df.loc[
        comparison_df["model"] == "LightGBM + SMOTE (Production)", "macro_f1"
    ].values[0]
    assert prod_f1 >= 0.85


def test_feature_importance_comparison_returns_dict():
    result = get_feature_importance_comparison()
    assert isinstance(result, dict)
    assert len(result) >= 1


def test_feature_importance_values_are_series():
    result = get_feature_importance_comparison()
    for label, importances in result.items():
        assert isinstance(importances, pd.Series)
        assert len(importances) > 0


def test_feature_importance_values_are_non_negative():
    """Feature importances should never be negative."""
    result = get_feature_importance_comparison()
    for label, importances in result.items():
        assert (importances >= 0).all(), f"{label} has negative importance values"
