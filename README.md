# 📊 Customer Churn Prediction System

A full **Machine Learning application** that predicts whether a customer is likely to churn.
The project demonstrates an **end-to-end ML workflow**, including data preprocessing, model training, API development, containerization, CI pipeline, and cloud deployment.

This repository showcases a **production-style ML system architecture** using **FastAPI, Streamlit, Docker, GitHub Actions, and cloud deployment**.

---

# 🌐 Live Application

**Streamlit Web App (Frontend)**
https://churn-streamlit-1b0f.onrender.com/

**FastAPI Prediction API (Backend)**
https://churn-api-h1cl.onrender.com/docs

The Streamlit interface sends customer information to the FastAPI API, which loads the trained model and returns a churn prediction.

---

# 🧠 Project Overview

Customer churn prediction helps companies identify customers who are likely to stop using their services. This project builds a machine learning pipeline to:

1. Process and clean customer data
2. Engineer predictive features
3. Train a classification model
4. Serve predictions via an API
5. Provide an interactive UI for predictions
6. Deploy the system in the cloud

---

# 🏗 System Architecture

```
User
  │
  ▼
Streamlit Web App
  │
  ▼
FastAPI Backend
  │
  ▼
Machine Learning Model
  │
  ▼
Prediction Response
```

Deployment architecture:

```
GitHub Repository
      │
      ▼
GitHub Actions (CI)
- Install dependencies
- Train model
- Build Docker containers
- Validate system
      │
      ▼
Render Cloud Deployment
      │
      ├── Streamlit Service
      └── FastAPI Service
```

---

# 📁 Project Structure

```
churn_prediction_system
│
├── app
│   └── streamlit_app.py          # Streamlit UI
│
├── src
│   ├── api
│   │   └── app.py                # FastAPI application
│   │
│   ├── models
│   │   └── train_model.py        # Model training pipeline
│   │
│   └── features                  # Feature engineering scripts
│
├── models
│   └── churn_model.pkl           # Trained model used in deployment
│
├── docker
│   ├── Dockerfile.api            # Container for FastAPI
│   ├── Dockerfile.streamlit      # Container for Streamlit
│   └── docker-compose.yml        # Local development containers
│
├── notebooks                     # EDA and experimentation
│
├── .github
│   └── workflows
│       └── ci.yml                # CI pipeline
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Technologies Used

* Python
* Scikit-learn
* Pandas
* FastAPI
* Streamlit
* Docker
* Docker Compose
* GitHub Actions (CI)
* Render (Cloud Deployment)

---

# 🧪 Running the Project Locally

## 1️⃣ Clone the repository

```
git clone https://github.com/NeejiMed/churn_prediction_system.git
cd churn_prediction_system
```

---

## 2️⃣ Create a virtual environment

```
python -m venv venv
source venv/bin/activate
```

Windows:

```
venv\Scripts\activate
```

---

## 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

# 🚀 Running the Application

## Option 1 — Run with Docker (Recommended)

```
cd docker
docker compose up --build
```

Services will start:

FastAPI API
http://localhost:8000/docs

Streamlit App
http://localhost:8501

---

## Option 2 — Run Manually

### Start API

```
uvicorn src.api.app:app --reload
```

### Start Streamlit

```
streamlit run app/streamlit_app.py
```

---

# 🤖 Model Training

To retrain the churn model:

```
python src/models/train_model.py
```

The trained model will be saved to:

```
models/churn_model.pkl
```

---

# 🔄 Continuous Integration (CI)

The project uses **GitHub Actions** to automate testing and validation.

Pipeline steps:

1. Checkout repository
2. Setup Python environment
3. Install dependencies
4. Train the churn model
5. Build Docker containers
6. Verify container startup

CI runs on pushes to:

```
main
dev
```

CI ensures the system builds successfully before deployment.

---

# 🚀 Deployment

The application is deployed using **Render cloud platform**.

Two services are deployed:

### FastAPI Service

Handles model inference.

```
https://churn-api-h1cl.onrender.com/docs
```

### Streamlit Service

User interface for predictions.

```
https://churn-streamlit-1b0f.onrender.com
```

Deployment is triggered automatically when changes are pushed to the **main branch**.

---

# 📊 Example API Request

```
POST /predict
```

Example JSON payload:

```
{
  "tenure": 12,
  "monthly_charges": 70.5,
  "total_charges": 840,
  "contract": "Month-to-month"
}
```

API response:

```
{
  "churn_prediction": 1,
  "probability": 0.82
}
```

---

# 🔧 Development Workflow

Branch strategy:

```
main        → production
dev         → development integration
feature/*   → new features
```

Typical workflow:

```
feature branch
      ↓
pull request
      ↓
dev
      ↓
main
      ↓
automatic deployment
```

---


# 📌 Future Improvements

Potential enhancements:

* Model monitoring
* Automated retraining
* Model registry (MLflow)
* Feature store
* Automated deployment pipeline

---

# 📜 License

This project currently does not include a license.
