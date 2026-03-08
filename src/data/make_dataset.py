import pandas as pd
from pathlib import Path

# Define the paths
RAW_DATA_PATH = Path("data/raw/customer_churn.csv")
PROCESSED_DATA_PATH = Path("data/processed/processed_customer_churn.csv")

def load_data():
    """Load the raw data from the specified path."""
    print('Loading raw dataset...')
    return pd.read_csv(RAW_DATA_PATH)

def clean_data(df):
    """Clean the data by handling missing values and encoding categorical variables."""
    print('Cleaning data...')
    # Remove spaces from column names
    df.columns = df.columns.str.strip()

    # Convert 'TotalCharges' to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill missing values in 'TotalCharges' with the median
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Encode target variable 'Churn' to binary
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df

def save_processed_data(df):
    """Save the processed data to the specified path."""
    print('Saving processed dataset...')
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    df.to_csv(PROCESSED_DATA_PATH, index=False)

def main():
    # Load the raw data
    df = load_data()

    # Clean the data
    df = clean_data(df)

    # Save the processed data
    save_processed_data(df)

    print('Data pipeline completed successfully.')

if __name__ == "__main__":
    main()