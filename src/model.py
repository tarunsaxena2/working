import pandas as pd
import numpy as np
import joblib
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

def build_pipeline(scale_pos_weight=None):
    """
    Build SMOTE + LightGBM imbalanced classification pipeline.
    SMOTE is applied only inside training folds to prevent data leakage.
    
    Parameters:
        scale_pos_weight (int): Weight for positive class (default=None)
    Returns:
        ImbPipeline: Pipeline with SMOTE and LGBMClassifier
    """
    smote = SMOTE(random_state=42)
    
    lgbm_params = {
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
        'n_estimators': 500,
        'learning_rate': 0.05,
        'num_leaves': 31
    }
    
    if scale_pos_weight is not None:
        lgbm_params['scale_pos_weight'] = scale_pos_weight
    
    lgbm = LGBMClassifier(**lgbm_params)
    
    pipeline = ImbPipeline([
        ('smote', smote),
        ('lgbm', lgbm)
    ])
    
    return pipeline

def save_model(pipeline, filepath='models/lgbm_pipeline.pkl'):
    """
    Save trained pipeline to disk using joblib.
    Note: models/ directory is excluded from git via .gitignore
    
    Parameters:
        pipeline: Trained pipeline object
        filepath (str): Path to save the model
    """
    import os
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath='models/lgbm_pipeline.pkl'):
    """
    Load trained pipeline from disk.
    
    Parameters:
        filepath (str): Path to load the model from
    Returns:
        pipeline: Loaded pipeline object
    """
    pipeline = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return pipeline

if __name__ == "__main__":
    pipeline = build_pipeline(scale_pos_weight=20)
    print("Best pipeline built successfully!")
    print(pipeline)

# ============================================
# MODEL SUMMARY — 
# ============================================
# build_pipeline() — SMOTE + LightGBM pipeline
# save_model() — saves trained model to models/
# load_model() — loads trained model from models/
# Best config: n_estimators=500, lr=0.05, num_leaves=31
# scale_pos_weight=20 (best from Day 3 tuning)
# ============================================

print("Model module loaded successfully")