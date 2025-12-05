import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
import shap
import pandas as pd

st.set_page_config(page_title="Heart Disease Predictor Premium", page_icon="❤️", layout="wide")

# ================================
# DARK MODE / UI STYLING
# ================================
toggle_dark = st.sidebar.checkbox("🌙 Dark Mode")

if toggle_dark:
    bg_color = "#0e1117"
    text_color = "white"
else:
    bg_color = "#ffffff"
    text_color = "black"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ================================
# PAGE HEADER
# ================================
st.markdown(
    f"""
    <h1 style='text-align: center; color:#ff4b4b;'>❤️ Heart Disease Prediction (Ultra Premium)</h1>
    <p style='text-align:center; font-size:18px; color:{text_color};'>
        AI-based Medical Risk Assessment Dashboard
    </p>
    """,
    unsafe_allow_html=True
)

# ================================
# LOAD MODEL
# ================================
model = joblib.load("best_heart_chd_model.joblib")
scaler = joblib.load("scaler_chd.joblib")

def to_bin(x):
    return 1 if x in ("Yes", "Male") else 0

# ================================
# FORM INPUTS
# ================================
st.markdown("### 🧍 Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 20, 100, 50)
    male = st.selectbox("Gender", ["Male", "Female"])
    education = st.number_input("Education (1–4)", 1, 4, 1)

with col2:
    currentSmoker = st.selectbox("Current Smoker?", ["Yes","No"])
    cigsPerDay = st.number_input("Cigarettes per day", 0, 50, 0)
    BPMeds = st.selectbox("On BP medication?", ["Yes","No"])

with col3:
    diabetes = st.selectbox("Diabetes?", ["Yes","No"])
    sysBP = st.number_input("Systolic BP", 80, 250, 120)
    BMI = st.number_input("BMI", 10.0, 60.0, 25.0)

col4, col5, col6 = st.columns(3)

with col4:
    totChol = st.number_input("Total Cholesterol", 100, 600, 200)

with col5:
    diaBP = st.number_input("Diastolic BP", 40, 150, 80)

with col6:
    glucose = st.number_input("Glucose", 40, 300, 90)

st.markdown("---")
center = st.columns([3,2,3])[1]
predict_btn = center.button("🚀 Predict Risk", use_container_width=True)

# ================================
# PREDICTION
# ================================
if predict_btn:

    input_data = np.array([[
        to_bin(male), age, education, to_bin(currentSmoker),
        cigsPerDay, to_bin(BPMeds), 0, 0, to_bin(diabetes),
        totChol, sysBP, diaBP, BMI, 70, glucose
    ]])

    scaled = scaler.transform(input_data)
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    # ============================
    # EXTREME FEATURE 1: RISK METER (GAUGE)
    # ============================
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Risk Percentage"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "red"},
            ],
        }
    ))

    st.markdown("## 🎯 Risk Level Gauge")
    st.plotly_chart(gauge, use_container_width=True)

    # ============================
    # EXTREME FEATURE 2: PROBABILITY BAR CHART
    # ============================
    st.markdown("### 📊 Probability Graph")

    bar = go.Figure(data=[
        go.Bar(name="Risk Probability", x=["CHD Risk"], y=[probability], marker_color="red")
    ])
    bar.update_layout(yaxis=dict(range=[0,1]))

    st.plotly_chart(bar, use_container_width=True)

    # ============================
    # TEXTUAL RESULT
    # ============================
    if prediction == 1:
        st.error(f"🚨 HIGH RISK — {probability:.2f}")
        st.snow()
    else:
        st.success(f"🟢 LOW RISK — {probability:.2f}")
        st.balloons()

