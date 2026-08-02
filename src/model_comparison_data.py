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
    """Run 5-fold Stratified CV with a Random Forest on the given feature set,
    and also fit once on full data to extract feature importances."""
    X = df[feature_list]
    y = df[y_col]

    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_validate(
        estimator=rf, X=X, y=y, cv=cv,
        scoring=["f1_macro", "precision_macro", "recall_macro"]
    )

    # Fit once on full data just to extract feature importances (not used for scoring)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=feature_list).sort_values(ascending=False)

    return {
        "macro_f1": round(scores["test_f1_macro"].mean(), 4),
        "precision": round(scores["test_precision_macro"].mean(), 4),
        "recall": round(scores["test_recall_macro"].mean(), 4),
        "feature_importances": importances,
    }


def compute_ablation_results():
    """
    Freshly re-runs the ablation study: Random Forest with vs. without
    external contextual features. Returns a list of result dicts
    (each including a 'feature_importances' Series for chart building).
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


def get_feature_importance_comparison():
    """
    Returns a dict of {label: pd.Series(feature_importances)} for the
    'With External Features' Random Forest run, plus the production
    LightGBM model's feature importances (loaded from the saved pipeline).

    Used by the Model Comparison dashboard page to explain *why* one
    model/feature-set wins, not just its score.
    """
    df = _load_engineered_data()
    rf_result = _evaluate_feature_set(df, ext_features)

    result = {
        "Random Forest (With External Features)": rf_result["feature_importances"]
    }

    # Add the production LightGBM model's feature importances
    try:
        import joblib
        pipeline = joblib.load("models/lgbm_retrained.pkl")
        lgbm_model = pipeline.named_steps.get("lgbm", pipeline)
        lgbm_features = [
            "Air_temperature_K", "Process_temperature_K", "Rotational_speed_rpm",
            "Torque_Nm", "Tool_wear_min", "Type_enc",
            "ambient_temp_C", "factory_load_pct", "humidity_pct"
        ]
        lgbm_importances = pd.Series(
            lgbm_model.feature_importances_, index=lgbm_features
        ).sort_values(ascending=False)
        result["LightGBM + SMOTE (Production)"] = lgbm_importances
    except Exception as e:
        print(f"Could not load production model importances: {e}")

    return result


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
    df = pd.DataFrame(all_results)
    # Drop the feature_importances column here — it's a Series, not scalar,
    # and is available separately via get_feature_importance_comparison()
    if "feature_importances" in df.columns:
        df = df.drop(columns=["feature_importances"])
    return df

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
