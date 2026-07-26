import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from lightgbm import LGBMClassifier

df = pd.read_csv("data/fused_dataset.csv")

print("\nDataset Loaded Successfully")

print("Shape:", df.shape)

target_col = "Machine failure"

X = df.drop(columns=[target_col])

# Remove ID column
if "Product ID" in X.columns:
    X = X.drop(columns=["Product ID"])

# Encode machine type
if "Type" in X.columns:
    X = pd.get_dummies(
        X,
        columns=["Type"],
        drop_first=True
    )

y = df[target_col]

# Clean feature names
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

# Hyperparameter Grid

n_estimators_list = [200, 500, 800]

learning_rate_list = [0.01, 0.05, 0.1]

num_leaves_list = [15, 31, 63]

print("\nHyperparameter Grid Ready")

print("n_estimators :", n_estimators_list)

print("learning_rate :", learning_rate_list)

print("num_leaves :", num_leaves_list)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nCross Validation Ready")

def evaluate_model(
    n_estimators,
    learning_rate,
    num_leaves
):

    model = LGBMClassifier(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    num_leaves=num_leaves,
    random_state=42,
    verbose=-1
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
        scoring=["f1_macro"],
        n_jobs=-1
    )

    mean_f1 = scores["test_f1_macro"].mean()

    return mean_f1
results = []

print("\nRunning Hyperparameter Search...\n")

for n_estimators in n_estimators_list:

    for learning_rate in learning_rate_list:

        for num_leaves in num_leaves_list:

            score = evaluate_model(
                n_estimators,
                learning_rate,
                num_leaves
            )

            results.append({
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "num_leaves": num_leaves,
                "f1_macro": score
            })

            print(
                f"Trees={n_estimators} | "
                f"LR={learning_rate} | "
                f"Leaves={num_leaves} | "
                f"F1={score:.4f}"
            )

        results_df = pd.DataFrame(results)

print("\nTop Results:")
print(
    results_df.sort_values(
        by="f1_macro",
        ascending=False
    ).head(10)
)
best_result = results_df.sort_values(
    by="f1_macro",
    ascending=False
).iloc[0]

print("\nBest Configuration:")

print(best_result)

results_df.to_csv(
    "data/hyperparameter_results.csv",
    index=False
)

print("\nResults Saved Successfully")

print("\nTotal Combinations Tested:")
print(len(results_df))

print("Best Model Ready For Next Stage")
