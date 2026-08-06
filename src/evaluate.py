from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


def evaluate_model(X, y, model):
    """
    Evaluate a classification model using stratified cross-validation
    with SMOTE applied inside each fold.

    Parameters:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        model: an unfitted sklearn/LightGBM classifier (SMOTE + model pipeline built internally)

    Returns:
        dict: Dictionary containing mean f1/precision/recall and std of f1
    """
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

print("Evaluate module loaded successfully")
