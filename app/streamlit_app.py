import streamlit as st
import pandas as pd
import joblib

artifact = joblib.load("models/churn_model.pkl")
model = artifact['model']
features = artifact['features']

st.title("Customer Churn Prediction")

tenure = st.slider("Tenure (months)", 0, 72)
monthly_charges = st.number_input("Monthly Charges")
total_charges = st.number_input("Total Charges")

input_data = pd.DataFrame(columns=features)
input_data.loc[0] = 0
input_data['tenure'] = tenure
input_data['MonthlyCharges'] = monthly_charges
input_data['TotalCharges'] = total_charges

if st.button("Predict Churn"):

    probability = model.predict_proba(input_data)[0][1]
    st.write(f"Predicted Probability of Churn: {probability:.2f}")
    print("probability: ", probability)
    if probability > 0.4:
        st.warning("⚠ High Churn Risk! Consider taking action to retain this customer.")
    else:
        st.success("✅ Low Churn Risk! This customer is likely to stay.")
        