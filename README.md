# 📈 Churn Prediction System

This repository houses a comprehensive system for predicting customer churn. It encompasses data preprocessing, feature engineering, model training, and deployment readiness.

## 🌟 Features

*   **Data Preprocessing:** Scripts to clean and prepare raw customer data.
*   **Feature Engineering:** Techniques to create informative features for churn prediction.
*   **Model Training:** Implementation of various machine learning models for churn classification.
*   **Model Evaluation:** Tools to compare and evaluate model performance.
*   **Threshold Optimization:** Methods to determine optimal prediction thresholds.
*   **API Deployment Readiness:** Components structured for API deployment.
*   **Streamlit Application:** A user-friendly interface for interacting with the prediction model.
*   **Dockerization:** Dockerfiles for containerizing the application and API.

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/NeejiMed/churn_prediction_system.git
    cd churn_prediction_system
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 💡 Usage

### Running the Streamlit Application

To launch the interactive Streamlit application, navigate to the `app` directory and run:

```bash
cd app
streamlit run streamlit_app.py
```

This will open the application in your web browser, allowing you to explore data and potentially make predictions.

### Running the API (Dockerized)

To run the prediction API using Docker:

1.  **Build the Docker images:**
    ```bash
    docker-compose build
    ```

2.  **Run the Docker containers:**
    ```bash
    docker-compose up
    ```

The API will be accessible at the port specified in `docker-compose.yml`. You can then send requests to this API to get churn predictions.

### Running Notebooks

The `notebooks` directory contains Jupyter notebooks for data exploration and experimentation. You can open and run these using your preferred Jupyter environment:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## 🤝 Contributing

We welcome contributions to improve the churn prediction system. Please follow these guidelines:

1.  **Fork the repository.**
2.  **Create a new branch** for your feature or bug fix.
3.  **Make your changes** and ensure they are well-tested.
4.  **Commit your changes** with clear and concise messages.
5.  **Push to your fork** and open a Pull Request.

Please ensure your code adheres to the existing style and includes relevant documentation.

## 📜 License

This project does not have a specified license.

---
