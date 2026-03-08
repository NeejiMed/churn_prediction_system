import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Define the paths
DATA_PATH = Path("data/processed")

def load_data():
    """Load the training and testing data from the specified paths."""
    print('Loading training and testing datasets...')
    X_train = pd.read_csv(DATA_PATH / "X_train.csv")
    X_test = pd.read_csv(DATA_PATH / "X_test.csv")
    y_train = pd.read_csv(DATA_PATH / "y_train.csv")
    y_test = pd.read_csv(DATA_PATH / "y_test.csv")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Train a logistic regression model on the training data."""
    print('Training the logistic regression model...')
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')  # Increased max_iter to ensure convergence
    model.fit(X_train, y_train.values.ravel())  # Use .values.ravel() to convert DataFrame to 1D array
    
    return model

def make_predictions(model, X_test):
    """Make predictions using the trained model on the test data."""
    print('Making predictions on the test dataset...')
    predictions = model.predict(X_test)
    return predictions

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance using classification report, confusion matrix, and accuracy score."""
    print('Evaluating the model...')
    predictions = make_predictions(model, X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    
    print("Accuracy Score:")
    print(accuracy_score(y_test, predictions))

def main():
    # Load the training and testing data
    X_train, X_test, y_train, y_test = load_data()

    # Train the logistic regression model
    model = train_model(X_train, y_train)

    y_pred = make_predictions(model, X_test)

    # Evaluate the model's performance
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()