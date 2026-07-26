import pandas as pd
import numpy as np

# =====================================================
# Sensor Columns
# =====================================================

SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# =====================================================
# Sort and Reset Index
# =====================================================

def sort_and_reset(df):
    """
    Sort DataFrame by index and reset index.
    """
    df = df.sort_index()
    df = df.reset_index(drop=True)
    return df


# =====================================================
# Rolling Mean
# =====================================================

def add_rolling_mean(df, window=5):

    for col in SENSOR_COLUMNS:
        df[f"{col}_rolling_mean"] = df[col].rolling(window).mean()

    return df


# =====================================================
# Rolling Standard Deviation
# =====================================================

def add_rolling_std(df, window=5):

    for col in SENSOR_COLUMNS:
        df[f"{col}_rolling_std"] = df[col].rolling(window).std()

    return df


# =====================================================
# Rolling Variance
# =====================================================

def add_rolling_var(df, window=5):

    for col in SENSOR_COLUMNS:
        df[f"{col}_rolling_var"] = df[col].rolling(window).var()

    return df


# =====================================================
# Complete Rolling Feature Pipeline
# =====================================================

def generate_rolling_features(df, window=5):
    """
    Generate rolling mean, std and variance
    for all sensor columns.
    """

    df = add_rolling_mean(df, window)
    df = add_rolling_std(df, window)
    df = add_rolling_var(df, window)

    df = df.dropna()
    df = df.reset_index(drop=True)

    rolling_features = [c for c in df.columns if "rolling" in c]

    print(f"Output Shape : {df.shape}")
    print(f"Rolling Features Added : {len(rolling_features)}")

    return df


# =====================================================
# External Context Merge
# =====================================================

def merge_external_context(df):
    """
    Simulate external environmental features.
    """

    np.random.seed(42)

    cols_before = len(df.columns)

    df["ambient_temp_C"] = np.random.normal(
        loc=28,
        scale=5,
        size=len(df)
    )

    df["factory_load_pct"] = np.random.uniform(
        50,
        100,
        size=len(df)
    )

    df["humidity_pct"] = np.random.normal(
        loc=60,
        scale=10,
        size=len(df)
    )

    cols_after = len(df.columns)

    print(f"Columns Before : {cols_before}")
    print(f"Columns After  : {cols_after}")
    print(f"Added Columns  : {cols_after-cols_before}")

    return df


# =====================================================
# Test
# =====================================================

if __name__ == "__main__":

    df = pd.read_csv("data/ai4i2020.csv")

    df = sort_and_reset(df)

    df = generate_rolling_features(df)

    df = merge_external_context(df)

    print("Feature Engineering Pipeline Completed Successfully.")

# ============================================
# MODULE SUMMARY — FINAL VERSION
# ============================================
# 1. SENSOR_COLUMNS — standardized list of all 5 sensor features
# 2. sort_and_reset() — sorts and resets DataFrame index
# 3. generate_rolling_features() — rolling mean, std, variance
# 4. merge_external_context() — adds ambient_temp, factory_load, humidity
#
# Week 1: Rolling features implemented ✅
# Week 2: External context fusion implemented ✅
# Week 3: Used in LightGBM + SMOTE pipeline ✅
# Week 4: Used in retrain.py end-to-end pipeline ✅
# Project Status: COMPLETE ✅
# ============================================

print("Feature engineering module loaded successfully")