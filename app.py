import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------- Load Saved Model, Scaler, and Feature List ----------
model = pickle.load(open("insurance_model.pkl", "rb"))
scaler = pickle.load(open("insurance_scaler.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))  # List of all features used in training

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Insurance Claim Predictor", layout="centered")
st.title("🛡️ Insurance Claim Approval Prediction")
st.markdown("Predict whether an insurance claim will be approved or not based on customer & incident details.")

st.sidebar.header("Enter Customer Information")

# ---------- Input Form ----------
def user_input_features():
    age = st.sidebar.slider("Age", 18, 85, 35)
    months_as_customer = st.sidebar.slider("Months as Customer", 0, 300, 100)
    policy_deductable = st.sidebar.selectbox("Policy Deductible", [500, 1000, 2000])
    annual_premium = st.sidebar.slider("Annual Premium", 200, 2000, 900)
    umbrella_limit = st.sidebar.selectbox("Umbrella Limit", [0, 600000, 1200000])
    capital_gains = st.sidebar.slider("Capital Gains", 0, 50000, 0)
    capital_loss = st.sidebar.slider("Capital Loss", 0, 50000, 0)
    incident_hour = st.sidebar.slider("Incident Hour", 0, 23, 12)
    total_claim_amount = st.sidebar.slider("Total Claim Amount", 0, 100000, 15000)
    injury_claim = st.sidebar.slider("Injury Claim", 0, 100000, 10000)
    property_claim = st.sidebar.slider("Property Claim", 0, 100000, 10000)
    vehicle_claim = st.sidebar.slider("Vehicle Claim", 0, 100000, 10000)
    witnesses = st.sidebar.slider("Number of Witnesses", 0, 5, 0)
    bodily_injuries = st.sidebar.slider("Bodily Injuries", 0, 2, 0)
    num_vehicles = st.sidebar.slider("Vehicles Involved", 1, 4, 1)

    # Prepare partial dataframe with only user-controlled features
    input_data = {
        'months_as_customer': months_as_customer,
        'age': age,
        'policy_deductable': policy_deductable,
        'policy_annual_premium': annual_premium,
        'umbrella_limit': umbrella_limit,
        'capital-gains': capital_gains,
        'capital-loss': capital_loss,
        'incident_hour_of_the_day': incident_hour,
        'total_claim_amount': total_claim_amount,
        'injury_claim': injury_claim,
        'property_claim': property_claim,
        'vehicle_claim': vehicle_claim,
        'witnesses': witnesses,
        'bodily_injuries': bodily_injuries,
        'number_of_vehicles_involved': num_vehicles
    }

    df = pd.DataFrame([input_data])
    return df

# ---------- User Input ----------
user_df = user_input_features()

# ---------- Ensure All Required Features ----------
# Fill missing features with 0
for col in feature_names:
    if col not in user_df.columns:
        user_df[col] = 0

# Reorder columns to match training data
user_df = user_df[feature_names]

# ---------- Predict ----------
scaled_input = scaler.transform(user_df)
prediction = model.predict(scaled_input)[0]

# ---------- Output ----------
st.subheader("Prediction Result")
if prediction == 1:
    st.success("✅ The claim is likely to be APPROVED.")
else:
    st.error("❌ The claim is likely to be DENIED.")

with st.expander("🔍 Show Processed Input Data"):
    st.write(user_df)
