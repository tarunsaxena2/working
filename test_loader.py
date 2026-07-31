from src.data_loader import load_data

df = load_data("data/cleaned_ai4i.csv")

print(df.shape)
print(df.head())
from src.data_loader import load_data, clean_data

df = load_data("data/cleaned_ai4i.csv")

print("Original Shape:")
print(df.shape)

clean_df = clean_data(df)

print("\nCleaned Shape:")
print(clean_df.shape)