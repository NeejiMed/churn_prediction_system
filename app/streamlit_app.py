import streamlit as st
import pandas as pd
import requests

st.title("Customer Churn Prediction")

tenure = st.slider("Tenure (months)", 0, 72)
monthly_charges = st.number_input("Monthly Charges")
total_charges = st.number_input("Total Charges")

if st.button("Predict Churn"):

    input_data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    response = requests.post(
        "http://api:8000/predict",
        json=input_data
    )

    result = response.json()
    probability = result["churn_probability"]

    st.write(f"Predicted Probability of Churn: {probability:.2f}")

    if probability > 0.4:
        st.warning("⚠ High Churn Risk! Consider taking action to retain this customer.")
    else:
        st.success("✅ Low Churn Risk! This customer is likely to stay.")