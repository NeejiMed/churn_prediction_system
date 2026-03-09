import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler # It scales numeric features so models learn better
from sklearn.model_selection import train_test_split

PROCESSED_DATA_PATH = Path("data/processed/processed_customer_churn.csv")
FEATURE_DATA_PATH = Path("data/processed/feature_data.csv")

def load_data():
    """Load the processed data from the specified path."""
    print('Loading processed dataset...')
    return pd.read_csv(PROCESSED_DATA_PATH)

def separate_features_target(df):
    """Separate features and target variable."""
    print('Separating features and target variable...')
    X = df.drop(columns=['Churn', 'customerID'])  # Drop target and non-informative columns
    y = df['Churn']
    return X, y

def encode_categorical_features(X):
    """Encode categorical features using one-hot encoding."""
    print('Encoding categorical features...')
    X_encoded = pd.get_dummies(X, drop_first=True)  # Drop first to avoid multicollinearity
    return X_encoded

def scale_features(X):
    """Scale numeric features using StandardScaler."""
    print('Scaling numeric features...')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)  # Convert back to DataFrame for easier handling
    return X_scaled

def split_data(X, y):
    """Split the data into training and testing sets."""
    print('Splitting data into training and testing sets...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Stratify to maintain class distribution - Because our dataset is imbalanced

    return X_train, X_test, y_train, y_test

def save_feature_data(X_train, X_test, y_train, y_test):
    """Save the feature data to the specified path."""
    print('Saving feature dataset...')
    
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

def main():
    # Load the processed data
    df = load_data()

    # Separate features and target variable
    X, y = separate_features_target(df)

    # Encode categorical features
    X_encoded = encode_categorical_features(X)

    # Scale numeric features
    X_scaled = scale_features(X_encoded)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    # Save the feature data
    save_feature_data(X_train, X_test, y_train, y_test)

    print('Feature engineering pipeline completed successfully.')

if __name__ == "__main__":
    main()