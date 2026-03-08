import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

DATA_PATH = Path("data/processed")
X_train = pd.read_csv(DATA_PATH / "X_train.csv")
y_train = pd.read_csv(DATA_PATH / "y_train.csv")

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train.values.ravel())

#feature importance
feature_importances = model.feature_importances_
features = X_train.columns

feat_importance = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
feat_importance = feat_importance.sort_values(by='Importance', ascending=False)

# Save feature importance to a CSV file
feat_importance = feat_importance.sort_values(by="importance", ascending=False)
