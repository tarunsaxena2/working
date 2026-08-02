"""
model_comparison_data.py — Ablation Study Results (Self-Contained)
Contextual Predictive Maintenance — IoT Edge AI

Re-runs the Week 2 ablation study (with-context vs without-context features)
fresh every time, instead of trusting stale/inconsistent notebook output.
Also includes the final production LightGBM model for cross-model comparison.

Usage:
    from src.model_comparison_data import get_model_comparison_data
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.feature_engineering import (
    sort_and_reset,
    generate_rolling_features,
    merge_external_context,
)
from src.feature_sets import base_features, ext_features

DATA_PATH = "data/ai4i2020.csv"

# Verified final production model metrics (Week 1-4, model_results.md)
# LightGBM + SMOTE, tuned config: n_estimators=500, learning_rate=0.1, num_leaves=15
PRODUCTION_MODEL_RESULTS = {
    "model": "LightGBM + SMOTE (Production)",
    "feature_set": "With External Features",
    "macro_f1": 0.8501,
    "precision": 0.8233,
    "recall": 0.8825,
}


def _load_engineered_data():
    """Load raw data and apply the exact same feature engineering as the notebook."""
    df = pd.read_csv(DATA_PATH)
    df = sort_and_reset(df)
    df = generate_rolling_features(df)   # adds rolling mean/std/var, drops NaN rows
    df = merge_external_context(df)      # adds ambient_temp_C, factory_load_pct, humidity_pct

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df["Type_enc"] = le.fit_transform(df["Type"])

    return df


def _evaluate_feature_set(df, feature_list, y_col="Machine failure"):
    """Run 5-fold Stratified CV with a Random Forest on the given feature set."""
    X = df[feature_list]
    y = df[y_col]

    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_validate(
        estimator=rf, X=X, y=y, cv=cv,
        scoring=["f1_macro", "precision_macro", "recall_macro"]
    )

    return {
        "macro_f1": round(scores["test_f1_macro"].mean(), 4),
        "precision": round(scores["test_precision_macro"].mean(), 4),
        "recall": round(scores["test_recall_macro"].mean(), 4),
    }


def compute_ablation_results():
    """
    Freshly re-runs the ablation study: Random Forest with vs. without
    external contextual features. Returns a list of result dicts.
    """
    df = _load_engineered_data()

    base_result = _evaluate_feature_set(df, base_features)
    ext_result = _evaluate_feature_set(df, ext_features)

    return [
        {
            "model": "Random Forest",
            "feature_set": "Without External Features",
            **base_result,
        },
        {
            "model": "Random Forest",
            "feature_set": "With External Features",
            **ext_result,
        },
    ]


def get_model_comparison_data():
    """
    Returns a DataFrame comparing:
    - Random Forest without external context
    - Random Forest with external context
    - Final production LightGBM + SMOTE model (with external context)

    This is the single source of truth for the Model Comparison dashboard page.
    """
    rf_results = compute_ablation_results()
    all_results = rf_results + [PRODUCTION_MODEL_RESULTS]
    return pd.DataFrame(all_results)


if __name__ == "__main__":
    print("=== Running fresh ablation study (this may take a few seconds) ===\n")
    comparison_df = get_model_comparison_data()
    print(comparison_df.to_string(index=False))

    base_f1 = comparison_df.loc[
        (comparison_df["model"] == "Random Forest") &
        (comparison_df["feature_set"] == "Without External Features"), "macro_f1"
    ].values[0]
    ext_f1 = comparison_df.loc[
        (comparison_df["model"] == "Random Forest") &
        (comparison_df["feature_set"] == "With External Features"), "macro_f1"
    ].values[0]

    improvement = ((ext_f1 - base_f1) / base_f1) * 100
    print(f"\nF1 Improvement (external features, Random Forest): {improvement:+.2f}%")
