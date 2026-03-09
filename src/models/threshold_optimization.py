import pandas as pd
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

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

def train_xgboost(X_train, y_train):
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

def optimize_threshold(model, X_test, y_test):
    print('Optimizing the classification threshold...')
    y_prob = model.predict_proba(X_test)[:, 1]  # Get predicted probabilities for the positive class

    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5]
    
    best_threshold = 0
    best_f1 = 0

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f'Threshold: {threshold:.2f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}')

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f'Best Threshold: {best_threshold:.2f} with F1 Score: {best_f1:.4f}\n')
    return best_threshold
def main():
    X_train, X_test, y_train, y_test = load_data()
    model = train_xgboost(X_train, y_train)
    best_threshold = optimize_threshold(model, X_test, y_test) 
    print(f'Final Best Threshold: {best_threshold:.2f}')

if __name__ == "__main__":
    main()