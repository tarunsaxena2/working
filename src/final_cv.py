import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from lightgbm import LGBMClassifier

print("Week 3 Day 4 Started")

# =====================================
# Load Dataset
# =====================================

df = pd.read_csv("data/fused_dataset.csv")

print("\nDataset Loaded Successfully")
print("Shape:", df.shape)

print("\nColumns:")

for col in df.columns:
    print(col)

# =====================================
# Feature Preparation
# =====================================

target_col = "Machine failure"

X = df.drop(columns=[target_col])

# Remove Product ID
if "Product ID" in X.columns:
    X = X.drop(columns=["Product ID"])

# Convert Type into numeric columns
if "Type" in X.columns:
    X = pd.get_dummies(
        X,
        columns=["Type"],
        drop_first=True
    )

y = df[target_col]

# =====================================
# Clean Feature Names
# =====================================

X.columns = (
    X.columns
    .str.replace("[", "", regex=False)
    .str.replace("]", "", regex=False)
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
    .str.replace(" ", "_", regex=False)
    .str.replace("/", "_", regex=False)
)

print("\nFeature Shape:", X.shape)
print("Target Shape :", y.shape)

print("\nTarget Distribution:")
print(y.value_counts())

# =====================================
# Best Hyperparameters
# =====================================

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.1,
    num_leaves=15,
    random_state=42
)

print("\nBest Hyperparameters Loaded")

# =====================================
# Pipeline
# =====================================

pipeline = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("model", model)
])

print("Pipeline Created Successfully")

# =====================================
# Cross Validation
# =====================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("5-Fold Stratified CV Ready")

# =====================================
# Final Cross Validation
# =====================================

print("\nRunning Final Cross Validation...\n")

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

# =====================================
# Results
# =====================================

print("=" * 60)
print("FINAL CROSS VALIDATION RESULTS")
print("=" * 60)

print("\nMacro F1 Scores")
print(scores["test_f1_macro"])

print("\nPrecision Scores")
print(scores["test_precision_macro"])

print("\nRecall Scores")
print(scores["test_recall_macro"])

mean_f1 = scores["test_f1_macro"].mean()
std_f1 = scores["test_f1_macro"].std()

mean_precision = scores["test_precision_macro"].mean()
mean_recall = scores["test_recall_macro"].mean()

print("\nMean Macro F1 :", round(mean_f1, 4))
print("Std Dev       :", round(std_f1, 4))

print("Mean Precision:", round(mean_precision, 4))
print("Mean Recall   :", round(mean_recall, 4))

print("\nTarget Check")

if mean_f1 >= 0.85:
    print("SUCCESS: Macro F1 >= 0.85")
else:
    print("FAILED: Macro F1 < 0.85")
    print("Try tuning max_depth and min_child_samples.")
