from fastapi import FastAPI

app = FastAPI(title="Churn Prediction API")

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}