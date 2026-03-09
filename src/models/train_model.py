import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

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


def train_logistic_regression(X_train, y_train):
    """Train a logistic regression model on the training data."""
    print('Training the logistic regression model...')
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')  # Increased max_iter to ensure convergence
    model.fit(X_train, y_train.values.ravel())  # Use .values.ravel() to convert DataFrame to 1D array
    
    return model

# Train a random forest model for feature importance analysis
def train_random_forest(X_train, y_train):
    """Train a random forest classifier on the training data."""
    print('Training the random forest model...')
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train.values.ravel())
    
    return model

# Train an XGBoost model for feature importance analysis
def train_xgboost(X_train, y_train):
    """Train an XGBoost classifier on the training data."""
    print('Training the XGBoost model...')
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=3,  # Adjust this based on the imbalance ratio
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train.values.ravel())

    return model

def make_predictions(model, X_test):
    """Make predictions using the trained model on the test data."""
    print('Making predictions on the test dataset...')
    predictions = model.predict(X_test)
    return predictions

def evaluate_model(model, X_test, y_test):
    print('Evaluating the model...')
    predictions = make_predictions(model, X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    
    print("Accuracy Score:")
    print(accuracy_score(y_test, predictions))
 
    y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
    print("ROC AUC Score:")
    print(roc_auc_score(y_test, y_prob))

def main():
    # Load the training and testing data
    X_train, X_test, y_train, y_test = load_data()

    # Train the logistic regression model
    #model = train_logistic_regression(X_train, y_train)

    # Train the random forest model
    #model = train_random_forest(X_train, y_train)

    # Train the XGBoost model
    model = train_xgboost(X_train, y_train)

    y_pred = make_predictions(model, X_test)
    
    # Evaluate the model's performance
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()