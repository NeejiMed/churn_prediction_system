import pandas as pd
from pathlib import Path

import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# Define the paths
DATA_PATH = Path("data/processed")

def load_data():
    print('Loading training and testing datasets...')
    X_train = pd.read_csv(DATA_PATH / "X_train.csv")
    X_test = pd.read_csv(DATA_PATH / "X_test.csv")
    y_train = pd.read_csv(DATA_PATH / "y_train.csv")
    
    return X_train, X_test, y_train

def train_model(X_train, y_train):
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

def explain_model(model, X_train):
    print('Explaining the model using SHAP values...')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    return shap_values

def plot_feature_importance(shap_values, X_train):
    print('Plotting feature importance...')
    shap.summary_plot(shap_values, X_train, plot_type="bar")

    # Save the SHAP summary plot
    plt.savefig("reports/shap_feature_importance.png")

def main():
    X_train, X_test, y_train = load_data()
    model = train_model(X_train, y_train)
    shap_values = explain_model(model, X_train)
    plot_feature_importance(shap_values, X_train)

if __name__ == "__main__":
    main()