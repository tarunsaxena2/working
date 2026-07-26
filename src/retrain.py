"""
retrain.py — End-to-End Retraining Pipeline
Contextual Predictive Maintenance — IoT Edge AI
Member 3 — Context & Integration

Usage: python src/retrain.py
"""

import pandas as pd
import numpy as np
import re
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

def load_raw_data(filepath='data/ai4i2020.csv'):
    """
    Load raw AI4I dataset from CSV file.
    
    Parameters:
        filepath (str): Path to CSV file
    Returns:
        pd.DataFrame: Raw dataset
    """
    df = pd.read_csv(filepath)
    print(f"✅ Data loaded: {df.shape}")
    return df

def apply_feature_engineering(df):
    """
    Apply full feature engineering pipeline:
    - Encode Type column
    - Simulate external context signals
    
    Parameters:
        df (pd.DataFrame): Raw DataFrame
    Returns:
        tuple: (X, y, feature_names)
    """
    # Encode Type column
    le = LabelEncoder()
    df['Type_enc'] = le.fit_transform(df['Type'])
    
    # Simulate external context
    np.random.seed(42)
    df['ambient_temp_C'] = np.random.normal(loc=28, scale=5, size=len(df))
    df['factory_load_pct'] = np.random.uniform(50, 100, size=len(df))
    df['humidity_pct'] = np.random.normal(loc=60, scale=10, size=len(df))
    
    print("✅ External context signals added")
    
    # Define features
    ext_features = [
        'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
        'Type_enc', 'ambient_temp_C', 'factory_load_pct', 'humidity_pct'
    ]
    
    X = df[ext_features].copy()
    y = df['Machine failure']
    
    # Clean feature names for LightGBM
    X.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in X.columns]
    
    print(f"✅ Feature matrix: {X.shape}")
    print(f"✅ Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

def build_and_train(X, y):
    """
    Build SMOTE + LightGBM pipeline and train on full dataset.
    
    Parameters:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
    Returns:
        pipeline: Trained pipeline
    """
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build pipeline
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('lgbm', LGBMClassifier(
            random_state=42, n_jobs=-1, verbose=-1,
            n_estimators=500, learning_rate=0.05,
            num_leaves=31, scale_pos_weight=20
        ))
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    print("✅ Model trained successfully!")
    
    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    
    print(f"\n=== Evaluation Results ===")
    print(f"Macro F1:  {macro_f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    
    if macro_f1 >= 0.85:
        print("✅ Target KPI met! Macro F1 >= 0.85")
    else:
        print("⚠️ Target KPI not met. Consider further tuning.")
    
    return pipeline

def save_model(pipeline, filepath='models/lgbm_retrained.pkl'):
    """
    Save retrained model to disk.
    
    Parameters:
        pipeline: Trained pipeline
        filepath (str): Save path
    """
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"✅ Model saved to {filepath}")

if __name__ == "__main__":
    print("=== Starting Retraining Pipeline ===\n")
    
    # Step 1: Load raw data
    df = load_raw_data()
    
    # Step 2: Apply feature engineering
    X, y = apply_feature_engineering(df)
    
    # Step 3: Build and train model
    pipeline = build_and_train(X, y)
    
    # Step 4: Save model
    save_model(pipeline)
    
    print("\n=== Retraining Pipeline Complete! ===")

# ============================================
# RETRAIN PIPELINE SUMMARY
# ============================================
# 1. load_raw_data() — loads AI4I CSV
# 2. apply_feature_engineering() — encodes, simulates context
# 3. build_and_train() — SMOTE + LightGBM training
# 4. save_model() — saves to models/ (excluded from git)
# ============================================