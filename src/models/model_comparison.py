import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, roc_auc_score

# Define the paths
DATA_PATH = Path("data/processed")

def load_data():    
    print('Loading training and testing datasets...')

    X_train = pd.read_csv(DATA_PATH / "X_train.csv")
    X_test = pd.read_csv(DATA_PATH / "X_test.csv")
    y_train = pd.read_csv(DATA_PATH / "y_train.csv")
    y_test = pd.read_csv(DATA_PATH / "y_test.csv")
    
    return X_train, X_test, y_train, y_test

def get_models():
    
    models = {
        'Logistic Regression': 
            LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),

        'Random Forest':
            RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),

        'XGBoost':
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                scale_pos_weight=3,  # Adjust this based on the imbalance ratio
                random_state=42,
                eval_metric='logloss'
            )
    }

    return models

def evaluate_model(models, X_train, X_test, y_train, y_test):

    results = []

    for name, model in models.items():
        print(f'Training the {name} model...')
        model.fit(X_train, y_train.values.ravel())

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] 

        roc_auc = roc_auc_score(y_test, y_prob)

        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        results.append({
            'Model': name,
            'ROC AUC': roc_auc
        })

    return pd.DataFrame(results)

def main():
    X_train, X_test, y_train, y_test = load_data()
    models = get_models()
    results_df = evaluate_model(models, X_train, X_test, y_train, y_test)

    print("\nModel Comparison Results:")
    print(results_df.sort_values(by='ROC AUC', ascending=False))

if __name__ == "__main__":
    main()