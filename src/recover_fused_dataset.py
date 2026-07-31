import pandas as pd

from feature_engineering import (
    sort_and_reset,
    generate_rolling_features,
    merge_external_context
)

# Load dataset
df = pd.read_csv("data/ai4i2020.csv")

# Week 2 pipeline
df = sort_and_reset(df)

df = generate_rolling_features(df)

df = merge_external_context(df)

print(df.shape)

# Save fused dataset
df.to_csv(
    "data/fused_dataset.csv",
    index=False
)

print("fused_dataset.csv created successfully")