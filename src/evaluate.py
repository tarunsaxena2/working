
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


def evaluate_model(X, y, model):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    pipeline = Pipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", model)
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=[
            "f1_macro",
            "precision_macro",
            "recall_macro"
        ],
        n_jobs=-1
    )

    results = {
        "f1_macro": scores["test_f1_macro"].mean(),
        "precision_macro": scores["test_precision_macro"].mean(),
        "recall_macro": scores["test_recall_macro"].mean(),
        "std_dev": scores["test_f1_macro"].std()
    }

    return results
if __name__ == "__main__":

    import pandas as pd
    from lightgbm import LGBMClassifier

    df = pd.read_csv("data/fused_dataset.csv")

    target = "Machine failure"

    X = df.drop(columns=[target])

    if "Product ID" in X.columns:
        X = X.drop(columns=["Product ID"])

    if "Type" in X.columns:
        X = pd.get_dummies(
            X,
            columns=["Type"],
            drop_first=True
        )

    X.columns = (
        X.columns
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )

    y = df[target]

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.1,
        num_leaves=15,
        random_state=42
    )

    results = evaluate_model(X, y, model)

    print("\nEvaluation Results")
    print(results)

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score

def evaluate_model(pipeline, X, y, cv=5):
    """
    Evaluate a classification pipeline using stratified cross-validation.
    Returns a dictionary of all metrics.
    
    Parameters:
        pipeline: sklearn/imblearn Pipeline object
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        cv (int): Number of cross-validation folds (default=5)
    
    Returns:
        dict: Dictionary containing mean and std of all metrics
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    scoring = {
        'f1_macro': 'f1_macro',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro'
    }
    
    cv_results = cross_validate(
        pipeline, X, y,
        cv=skf,
        scoring=scoring,
        return_train_score=False
    )
    
    results = {
        'macro_f1_mean': cv_results['test_f1_macro'].mean(),
        'macro_f1_std': cv_results['test_f1_macro'].std(),
        'precision_mean': cv_results['test_precision_macro'].mean(),
        'precision_std': cv_results['test_precision_macro'].std(),
        'recall_mean': cv_results['test_recall_macro'].mean(),
        'recall_std': cv_results['test_recall_macro'].std(),
        'per_fold_f1': cv_results['test_f1_macro'].tolist()
    }
    
    print("=== Model Evaluation Results ===")
    print(f"Macro F1:  {results['macro_f1_mean']:.4f} (+/- {results['macro_f1_std']:.4f})")
    print(f"Precision: {results['precision_mean']:.4f} (+/- {results['precision_std']:.4f})")
    print(f"Recall:    {results['recall_mean']:.4f} (+/- {results['recall_std']:.4f})")
    print(f"Per-fold F1 scores: {[round(f, 4) for f in results['per_fold_f1']]}")
    
    if results['macro_f1_mean'] >= 0.85:
        print(f"\n✅ Target KPI met! Macro F1 >= 0.85")
    else:
        print(f"\n⚠️ Target KPI not met. Macro F1 < 0.85 — needs tuning")
    
    return results

# Unit test
if __name__ == "__main__":
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    from lightgbm import LGBMClassifier
    import pandas as pd
    import re
    from sklearn.preprocessing import LabelEncoder
    import numpy as np

    # Load data
    df = pd.read_csv("data/ai4i2020.csv")
    
    # Add external context
    np.random.seed(42)
    df['ambient_temp_C'] = np.random.normal(loc=28, scale=5, size=len(df))
    df['factory_load_pct'] = np.random.uniform(50, 100, size=len(df))
    df['humidity_pct'] = np.random.normal(loc=60, scale=10, size=len(df))
    
    le = LabelEncoder()
    df['Type_enc'] = le.fit_transform(df['Type'])
    
    ext_features = [
        'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
        'Type_enc', 'ambient_temp_C', 'factory_load_pct', 'humidity_pct'
    ]
    
    X = df[ext_features]
    y = df['Machine failure']
    
    # Clean feature names
    X.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in X.columns]
    
    # Build pipeline
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('lgbm', LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1))
    ])
    
    # Evaluate
    results = evaluate_model(pipeline, X, y, cv=5)
    print("\nUnit test passed!")

# ============================================
# EVALUATE MODULE SUMMARY
# ============================================
# evaluate_model(pipeline, X, y, cv=5)
# Returns: dict with macro_f1, precision, recall (mean + std)
# Target KPI: Macro F1 >= 0.85
# ============================================

print("Evaluate module loaded successfully")

