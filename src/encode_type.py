import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/ai4i2020.csv")

print("=== Before Encoding ===")
print(df["Type"].value_counts())

encoder = LabelEncoder()

df["Type_enc"] = encoder.fit_transform(df["Type"])

print("\n=== Encoding Mapping ===")
for original, encoded in zip(
        encoder.classes_,
        encoder.transform(encoder.classes_)):
    print(f"{original} -> {encoded}")

print("\n=== After Encoding ===")
print(df["Type_enc"].value_counts())

print("\nRows Before:", len(df))
print("Rows After :", len(df))

assert df["Type"].notnull().sum() == df["Type_enc"].notnull().sum()

print("✓ No Data Loss Detected")

print("\n=== Preview ===")
print(df[["Type", "Type_enc"]].head())

# Save encoded dataset
df.to_csv("data/cleaned_ai4i.csv", index=False)

print("\nFile saved successfully.")