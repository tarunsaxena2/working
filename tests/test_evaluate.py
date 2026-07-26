import pandas as pd

from lightgbm import LGBMClassifier

from src.evaluate import evaluate_model


# Load dataset
df = pd.read_csv("data/fused_dataset.csv")

target = "Machine failure"

X = df.drop(columns=[target])

# Remove Product ID
if "Product ID" in X.columns:
    X = X.drop(columns=["Product ID"])

# Encode Type
if "Type" in X.columns:
    X = pd.get_dummies(
        X,
        columns=["Type"],
        drop_first=True
    )

# Clean column names
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

results = evaluate_model(
    X,
    y,
    model
)

print(results)

assert "f1_macro" in results
assert "precision_macro" in results
assert "recall_macro" in results
assert "std_dev" in results

print("\nUnit Test Passed Successfully")