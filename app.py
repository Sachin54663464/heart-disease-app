import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details and predict 10-year CHD risk.")

# Load model + scaler
model = joblib.load("best_heart_chd_model.joblib")
scaler = joblib.load("scaler_chd.joblib")

def to_bin(x):
    return 1 if x in ("Yes","Male") else 0

# Input fields
age = st.number_input("Age", 20, 100, 50)
male = st.selectbox("Gender", ["Male", "Female"])
education = st.number_input("Education (1–4)", 1, 4, 1)
currentSmoker = st.selectbox("Current Smoker?", ["Yes","No"])
cigsPerDay = st.number_input("Cigarettes per day", 0, 50, 0)
BPMeds = st.selectbox("On BP medication?", ["Yes","No"])
prevalentStroke = st.selectbox("Stroke history?", ["Yes","No"])
prevalentHyp = st.selectbox("Hypertension?", ["Yes","No"])
diabetes = st.selectbox("Diabetes?", ["Yes","No"])
totChol = st.number_input("Total Cholesterol", 100, 600, 200)
sysBP = st.number_input("Systolic BP", 80, 250, 120)
diaBP = st.number_input("Diastolic BP", 40, 150, 80)
BMI = st.number_input("BMI", 10.0, 60.0, 25.0)
heartRate = st.number_input("Heart Rate", 40, 200, 72)
glucose = st.number_input("Glucose", 40, 300, 90)

if st.button("Predict"):

    data = np.array([[ 
        to_bin(male), age, education, to_bin(currentSmoker), cigsPerDay,
        to_bin(BPMeds), to_bin(prevalentStroke), to_bin(prevalentHyp),
        to_bin(diabetes), totChol, sysBP, diaBP, BMI, heartRate, glucose
    ]])

    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    prob = model.predict_proba(data_scaled)[0][1]

    if prediction == 1:
        st.error(f"HIGH RISK (Probability: {prob:.2f})")
    else:
        st.success(f"LOW RISK (Probability: {prob:.2f})")

