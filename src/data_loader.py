import pandas as pd


def load_data(filepath):
    """
    Load AI4I Predictive Maintenance dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV dataset file.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """
    return pd.read_csv(filepath)


def clean_data(df):
    """
    Clean the dataset by removing duplicate rows and missing values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataset.
    """
    df = df.copy()
    df = df.drop_duplicates()
    df = df.dropna()

    return df


if __name__ == "__main__":

    # Change this path according to your project structure
    file_path = "data/cleaned_ai4i.csv"

    try:
        # Load dataset
        df = load_data(file_path)

        print("=" * 50)
        print("Dataset Loaded Successfully")
        print("=" * 50)

        print("\nFirst 5 Rows:")
        print(df.head())

        print("\nDataset Shape Before Cleaning:")
        print(df.shape)

        print("\nColumn Names:")
        print(df.columns.tolist())

        # Clean dataset
        cleaned_df = clean_data(df)

        print("\nDataset Cleaned Successfully")
        print("-" * 50)

        print("Shape After Cleaning:")
        print(cleaned_df.shape)

        print("\nMissing Values:")
        print(cleaned_df.isnull().sum())

        print("\nDuplicate Rows:")
        print(cleaned_df.duplicated().sum())

        print("\nData Types:")
        print(cleaned_df.dtypes)

        print("\nData Loader Module Working Successfully!")

    except FileNotFoundError:
        print(f"\nError: Dataset not found at '{file_path}'")
        print("Please check the dataset path.")

    except Exception as e:
        print(f"\nUnexpected Error: {e}")