from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

artifact = joblib.load("models/churn_model.pkl")
model = artifact["model"]
features = artifact["features"]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: dict):

    input_df = pd.DataFrame(columns=features)
    input_df.loc[0] = 0

    input_df["tenure"] = data["tenure"]
    input_df["MonthlyCharges"] = data["MonthlyCharges"]
    input_df["TotalCharges"] = data["TotalCharges"]

    probability = model.predict_proba(input_df)[0][1]

    return {"churn_probability": float(probability)}