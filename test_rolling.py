import pandas as pd
from src.feature_engineering import generate_rolling_features

df = pd.read_csv("data/cleaned_ai4i.csv")

new_df = generate_rolling_features(df)

new_df.to_csv("data/rolling_ai4i.csv", index=False)

print("Saved Successfully")
print(new_df.shape)