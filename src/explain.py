"""
explain.py — SHAP Explainability Helper
Contextual Predictive Maintenance — IoT Edge AI

Usage:
    from src.explain import explain_single
"""

import pandas as pd
import shap
import joblib

MODEL_PATH = 'models/lgbm_retrained.pkl'

_explainer_cache = {}

def get_explainer(pipeline):
    """
    Build (or retrieve cached) SHAP TreeExplainer for the LightGBM model
    inside the SMOTE pipeline.
    """
    if 'explainer' not in _explainer_cache:
        # Extract the trained LightGBM model from the imblearn pipeline
        lgbm_model = pipeline.named_steps['lgbm']
        _explainer_cache['explainer'] = shap.TreeExplainer(lgbm_model)
    return _explainer_cache['explainer']

def explain_single(input_dict, pipeline=None):
    """
    Compute SHAP values for a single sensor reading.

    Parameters:
        input_dict (dict): same format as predict_single() input
        pipeline: preloaded pipeline (optional, loads fresh if None)

    Returns:
        dict: {
            'feature_names': list of feature names,
            'shap_values': list of SHAP values (impact per feature),
            'base_value': float (expected/average model output),
            'feature_values': list of input values (for reference)
        }
    """
    if pipeline is None:
        pipeline = joblib.load(MODEL_PATH)

    explainer = get_explainer(pipeline)

    X = pd.DataFrame([input_dict])
    shap_values = explainer.shap_values(X)

    # For binary classification, shap_values may be a list [class0, class1]
    # We want the "failure" class (class 1) contribution
    if isinstance(shap_values, list):
        values_for_failure = shap_values[1][0]
        base_value = explainer.expected_value[1]
    else:
        values_for_failure = shap_values[0]
        base_value = explainer.expected_value

    return {
        'feature_names': list(X.columns),
        'shap_values': [float(v) for v in values_for_failure],
        'base_value': float(base_value),
        'feature_values': [float(v) for v in X.iloc[0].values]
    }

if __name__ == "__main__":
    # Quick sanity test
    sample = {
        'Air_temperature_K': 298.5,
        'Process_temperature_K': 308.7,
        'Rotational_speed_rpm': 1500,
        'Torque_Nm': 40.2,
        'Tool_wear_min': 10,
        'Type_enc': 1,
        'ambient_temp_C': 28.0,
        'factory_load_pct': 75.0,
        'humidity_pct': 60.0
    }
    result = explain_single(sample)
    print("=== SHAP Explanation Test ===")
    for name, val, feat_val in zip(result['feature_names'], result['shap_values'], result['feature_values']):
        print(f"{name}: value={feat_val:.2f}, SHAP impact={val:+.4f}")
    print(f"\nBase value (expected output): {result['base_value']:.4f}")
