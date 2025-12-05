# app.py — CHD Predictor v4.0 (Modern HealthTech Pro)
# Built by Sachin Ravi
# Requires: best_heart_chd_model.joblib, scaler_chd.joblib, feature_order.json (recommended)
# If those files are not present the app will attempt to train using framingham.csv

import os
import io
import json
import time
import base64
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from fpdf import FPDF

import streamlit as st
from streamlit.components.v1 import html as st_html

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

# optional shap
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------
# Config & paths
# ---------------------------
st.set_page_config(page_title="CHD Predictor — v4.0", layout="wide", initial_sidebar_state="expanded")
ROOT = Path.cwd()
MODEL_PATH = ROOT / "best_heart_chd_model.joblib"
SCALER_PATH = ROOT / "scaler_chd.joblib"
FEATURE_ORDER_PATH = ROOT / "feature_order.json"
CSV_PATH = ROOT / "framingham.csv"
RANDOM_STATE = 42

# ---------------------------
# Utility functions
# ---------------------------
def load_feature_order():
    if FEATURE_ORDER_PATH.exists():
        try:
            with open(FEATURE_ORDER_PATH, "r") as f:
                features = json.load(f)
            return features
        except Exception:
            pass
    # fallback: derive from CSV if exists
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH, nrows=1)
        cols = list(df.columns)
        if "TenYearCHD" in cols:
            cols.remove("TenYearCHD")
        else:
            cols = cols[:-1]
        return cols
    # default fallback
    return ["age","male","education","currentSmoker","cigsPerDay","BPMeds","prevalentStroke","prevalentHyp","diabetes","totChol","sysBP","diaBP","BMI","heartRate","glucose"]

def download_and_prepare_full_df():
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
    else:
        # fallback remote
        try:
            df = pd.read_csv("https://raw.githubusercontent.com/ishank-j/propublica-tutorials/main/heart.csv")
        except Exception:
            # synthetic fallback (shouldn't be used in production)
            np.random.seed(RANDOM_STATE)
            n = 1200
            df = pd.DataFrame({
                "age": np.random.randint(30,85,size=n),
                "male": np.random.randint(0,2,size=n),
                "education": np.random.randint(1,5,size=n),
                "currentSmoker": np.random.randint(0,2,size=n),
                "cigsPerDay": np.random.randint(0,40,size=n),
                "BPMeds": np.random.randint(0,2,size=n),
                "prevalentStroke": np.random.randint(0,2,size=n),
                "prevalentHyp": np.random.randint(0,2,size=n),
                "diabetes": np.random.randint(0,2,size=n),
                "totChol": np.random.randint(130,340,size=n),
                "sysBP": np.random.randint(100,190,size=n),
                "diaBP": np.random.randint(60,110,size=n),
                "BMI": np.round(np.random.uniform(18,38,size=n),1),
                "heartRate": np.random.randint(55,110,size=n),
                "glucose": np.random.randint(70,220,size=n),
                "TenYearCHD": np.random.randint(0,2,size=n)
            })
    # cleaning (matching training script)
    df = df.replace(["NA","?"," ", ""], np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(df.median(numeric_only=True))
    return df

def train_and_save_from_csv():
    df = download_and_prepare_full_df()
    target_col = "TenYearCHD" if "TenYearCHD" in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced")
    clf.fit(X_train_s, y_train)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    # save feature order used for training
    feat_order = list(X.columns)
    with open(FEATURE_ORDER_PATH, "w") as f:
        json.dump(feat_order, f, indent=2)
    return clf, scaler, feat_order, (X_test, y_test)

def load_model_scaler():
    # prefer loading saved artifacts
    if MODEL_PATH.exists() and SCALER_PATH.exists() and FEATURE_ORDER_PATH.exists():
        try:
            clf = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            with open(FEATURE_ORDER_PATH, "r") as f:
                features = json.load(f)
            # build a test_split from CSV for permutation importance later
            df = download_and_prepare_full_df()
            target_col = "TenYearCHD" if "TenYearCHD" in df.columns else df.columns[-1]
            X = df.drop(columns=[target_col])
            y = df[target_col].astype(int)
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.18, random_state=RANDOM_STATE, stratify=y)
            return clf, scaler, features, (X_test, y_test)
        except Exception as e:
            st.warning(f"Failed to load artifacts: {e}. Retraining now.")
    # retrain if artifacts missing or load failed
    return train_and_save_from_csv()

# ---------------------------
# Load / train
# ---------------------------
model, scaler, FEATURE_NAMES, TEST_SPLIT = load_model_scaler()

# ensure FEATURE_NAMES is list
if FEATURE_NAMES is None or not isinstance(FEATURE_NAMES, list) or len(FEATURE_NAMES) == 0:
    FEATURE_NAMES = load_feature_order()

# ---------------------------
# Align features function
# ---------------------------
def align_features(df: pd.DataFrame, feature_names: list):
    df = df.copy()
    for c in feature_names:
        if c not in df.columns:
            df[c] = 0.0
    df = df[feature_names]
    return df.astype(float)

# ---------------------------
# Predict helpers
# ---------------------------
def preprocess_and_predict(df_aligned: pd.DataFrame):
    Xs = scaler.transform(df_aligned)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(Xs)[:,1][0])
    else:
        return float(model.predict(Xs)[0])

def risk_level(p):
    if p < 0.15: return "Low"
    if p < 0.35: return "Moderate"
    if p < 0.7: return "High"
    return "Very High"

def compute_model_accuracy():
    try:
        if TEST_SPLIT:
            X_test, y_test = TEST_SPLIT
        else:
            df = download_and_prepare_full_df()
            target_col = "TenYearCHD" if "TenYearCHD" in df.columns else df.columns[-1]
            X = df.drop(columns=[target_col]); y = df[target_col].astype(int)
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.18, random_state=RANDOM_STATE, stratify=y)
        X_test_aligned = align_features(X_test, FEATURE_NAMES)
        Xs = scaler.transform(X_test_aligned)
        try:
            ypred = model.predict(Xs)
        except Exception:
            if hasattr(model, "predict_proba"):
                ypred = (model.predict_proba(Xs)[:,1] >= 0.5).astype(int)
            else:
                return None
        return float(accuracy_score(y_test, ypred))
    except Exception:
        return None

MODEL_ACCURACY = compute_model_accuracy()
ACC_STR = f"{MODEL_ACCURACY:.2%}" if MODEL_ACCURACY is not None else "N/A"

# ---------------------------
# UI style (clean, no-flicker)
# ---------------------------
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, "Helvetica Neue", Arial; }
    .card { background: white; border-radius:10px; padding:14px; box-shadow: 0 8px 30px rgba(15,23,42,0.06); }
    .title { font-weight:800; font-size:20px; }
    .muted { color:#6b7280; }
    .soft-divider { height:1px; background:linear-gradient(90deg, rgba(0,0,0,0.03), rgba(0,0,0,0.01)); margin:12px 0; border-radius:2px; }
    .counter { font-weight:800; font-size:28px; color:#05204a; }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ---------------------------
# Header + Sidebar
# ---------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

with st.sidebar:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## Settings")
    keep_bg = st.checkbox("Enable subtle background", value=False)
    st.markdown("---")
    preset = st.selectbox("Load preset", options=["Custom","Healthy demo","High-risk smoker","Elderly hypertensive"])
    if st.button("Clear history"):
        st.session_state["history"] = []
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(f"""
<div style="display:flex; align-items:center; justify-content:space-between;">
  <div>
    <div class="title">Clinical Heart Risk — v4.0</div>
    <div class="muted">Built by Sachin Ravi • Protected & aligned model</div>
  </div>
  <div class="muted">Model accuracy: <strong>{ACC_STR}</strong></div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Input form (vertical; keys stable)
# ---------------------------
c1, c2 = st.columns([1.6,1], gap="large")
with c1:
    with st.form("patient_form"):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Patient details")
        age = st.number_input("Age", min_value=18, max_value=120, value=58, key="age")
        gender = st.selectbox("Gender", options=["Male","Female"], index=0, key="gender")
        education = st.selectbox("Education (1-4)", options=[1,2,3,4], index=1, key="education")
        smoker = st.selectbox("Current smoker?", options=["Smoker","Non-smoker"], index=0, key="smoker")
        cigsPerDay = st.number_input("Cigarettes per day", min_value=0, max_value=100, value=20, key="cigsPerDay")
        BPMeds = st.selectbox("On BP medication?", options=["Yes","No"], index=0, key="BPMeds")
        prevalentStroke = st.selectbox("Stroke history?", options=["Yes","No"], index=0, key="prevalentStroke")
        prevalentHyp = st.selectbox("Hypertension?", options=["Yes","No"], index=0, key="prevalentHyp")
        diabetes = st.selectbox("Diabetes?", options=["Yes","No"], index=0, key="diabetes")
        totChol = st.number_input("Total cholesterol", min_value=100.0, max_value=600.0, value=250.0, key="totChol")
        sysBP = st.number_input("Systolic BP", min_value=80.0, max_value=260.0, value=160.0, key="sysBP")
        diaBP = st.number_input("Diastolic BP", min_value=40.0, max_value=160.0, value=95.0, key="diaBP")
        BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=29.5, key="BMI")
        heartRate = st.number_input("Heart Rate", min_value=30.0, max_value=200.0, value=90.0, key="heartRate")
        glucose = st.number_input("Glucose", min_value=40.0, max_value=400.0, value=140.0, key="glucose")
        st.markdown("</div>", unsafe_allow_html=True)
        submit = st.form_submit_button("Predict Risk")

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Quick actions")
    st.markdown(f"<div class='muted'>Model accuracy: <strong>{ACC_STR}</strong></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Map UI -> model features (fresh on each submit)
# ---------------------------
def build_input_from_widgets():
    mapping = {
        "age": float(age),
        "male": 1.0 if gender == "Male" else 0.0,
        "education": float(education),
        "currentSmoker": 1.0 if smoker == "Smoker" else 0.0,
        "cigsPerDay": float(cigsPerDay),
        "BPMeds": 1.0 if BPMeds == "Yes" else 0.0,
        "prevalentStroke": 1.0 if prevalentStroke == "Yes" else 0.0,
        "prevalentHyp": 1.0 if prevalentHyp == "Yes" else 0.0,
        "diabetes": 1.0 if diabetes == "Yes" else 0.0,
        "totChol": float(totChol),
        "sysBP": float(sysBP),
        "diaBP": float(diaBP),
        "BMI": float(BMI),
        "heartRate": float(heartRate),
        "glucose": float(glucose)
    }
    # Build ordered list following FEATURE_NAMES; if missing, default 0.0
    ordered = {fn: mapping.get(fn, 0.0) for fn in FEATURE_NAMES}
    return ordered

# ---------------------------
# Prediction & Explainability
# ---------------------------
if submit:
    inputs = build_input_from_widgets()
    X_user = pd.DataFrame([inputs])
    X_aligned = align_features(X_user, FEATURE_NAMES)

    # confirm the aligned df (debug - comment out in prod)
    # st.write("Model input (aligned):", X_aligned)

    try:
        prob = preprocess_and_predict(X_aligned)
        label = risk_level(prob)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # save history
    rec = {"ts": datetime.now().isoformat(timespec='seconds'), "inputs": inputs, "prob": prob, "label": label}
    st.session_state["history"].insert(0, rec)
    st.session_state["history"] = st.session_state["history"][:50]

    # Display results
    st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
    r1, r2 = st.columns([1.7,1], gap="large")
    with r1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div class='muted'>Risk probability</div>", unsafe_allow_html=True)
        # gauge
        fig = go.Figure(go.Indicator(mode="gauge+number", value=prob*100,
                                     gauge={'axis':{'range':[0,100]},
                                            'bar':{'color':'#ef4444'}}))
        fig.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown(f"<div style='text-align:center; font-weight:700; margin-top:6px;'>{prob*100:.1f}% — {label}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with r2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div class='muted'>Recommendation</div>", unsafe_allow_html=True)
        if prob > 0.7:
            st.markdown("<div style='font-weight:700;'>Urgent clinical review</div>", unsafe_allow_html=True)
            st.write("ECG, fasting glucose, lipid panel, cardiology consult.")
        elif prob > 0.35:
            st.markdown("<div style='font-weight:700;'>High risk — consider follow-up</div>", unsafe_allow_html=True)
            st.write("Lifestyle, BP control, further testing.")
        else:
            st.markdown("<div style='font-weight:700;'>Low risk — maintain screening</div>", unsafe_allow_html=True)
            st.write("Lifestyle, routine checks.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Explainability block
    st.markdown("<div style='margin-top:12px;' class='card'>", unsafe_allow_html=True)
    st.markdown("### Explainability")
    use_perm = False
    Xs = None
    try:
        Xs = scaler.transform(X_aligned)
    except Exception:
        Xs = None

    if SHAP_AVAILABLE and Xs is not None:
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(Xs)
            if isinstance(shap_vals, list):
                shap_single = shap_vals[1][0]
            else:
                shap_single = shap_vals[0]
            contribs = {FEATURE_NAMES[i]: float(shap_single[i]) for i in range(len(FEATURE_NAMES))}
            top = dict(sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6])
            chips = "".join([f"<div class='chip'><strong>{k}</strong>: {v:.2f}</div>" for k,v in top.items()])
            st.markdown(chips, unsafe_allow_html=True)
            df_bar = pd.DataFrame({"feature": list(top.keys()), "impact": [top[k] for k in top]})
            st.plotly_chart(px.bar(df_bar, x="impact", y="feature", orientation="h", height=260), use_container_width=True)
        except Exception:
            st.warning("SHAP failed — switching to permutation importance")
            use_perm = True
    else:
        if not SHAP_AVAILABLE:
            st.warning("SHAP not installed — using permutation importance fallback")
        use_perm = True

    if use_perm:
        try:
            from sklearn.inspection import permutation_importance
            if TEST_SPLIT is not None:
                X_test, y_test = TEST_SPLIT
                X_test_aligned = align_features(X_test, FEATURE_NAMES)
                X_batch = scaler.transform(X_test_aligned)
                y_batch = y_test.values
            else:
                df_all = download_and_prepare_full_df()
                target_col = "TenYearCHD" if "TenYearCHD" in df_all.columns else df_all.columns[-1]
                X_tmp = df_all.drop(columns=[target_col])
                X_tmp_aligned = align_features(X_tmp, FEATURE_NAMES)
                X_batch = scaler.transform(X_tmp_aligned)
                y_batch = df_all[target_col].astype(int).values

            res = permutation_importance(model, X_batch, y_batch, n_repeats=12, random_state=42)
            pi = res.importances_mean
            if len(pi) == len(FEATURE_NAMES):
                df_pi = pd.DataFrame({"feature": FEATURE_NAMES, "importance": pi}).sort_values("importance", ascending=False)
            else:
                df_pi = pd.DataFrame({"feature": [f"f{i}" for i in range(len(pi))], "importance": pi}).sort_values("importance", ascending=False)
            st.plotly_chart(px.bar(df_pi, x="importance", y="feature", orientation="h", height=260), use_container_width=True)
        except Exception as e:
            st.error(f"Permutation importance failed: {e}")
            st.info("Explainability unavailable for this model.")
    st.markdown("</div>", unsafe_allow_html=True)

    # PDF export
    def fig_to_bytes(fig, fmt="png"):
        try:
            return fig.to_image(format=fmt, scale=2)
        except Exception:
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            return buf.read()

    def build_pdf(inputs, prob, label):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14, style="B")
        pdf.cell(200, 8, "Clinical Heart Disease Risk Report", ln=True, align="C")
        pdf.ln(4)
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(6)
        pdf.set_font("Arial", size=12, style="B")
        pdf.cell(0, 6, "Inputs", ln=True)
        pdf.set_font("Arial", size=10)
        for k,v in inputs.items():
            pdf.cell(0, 6, f"{k}: {v}", ln=True)
        pdf.ln(6)
        pdf.set_font("Arial", size=12, style="B")
        pdf.cell(0, 6, "Prediction", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 6, f"Probability: {prob:.3f}", ln=True)
        pdf.cell(0, 6, f"Risk level: {label}", ln=True)
        try:
            figg = go.Figure(go.Indicator(mode="gauge+number", value=prob*100))
            img = fig_to_bytes(figg)
            tmp = "tmp_gauge.png"
            with open(tmp, "wb") as f:
                f.write(img)
            pdf.image(tmp, w=160)
        except Exception:
            pdf.cell(0, 6, "Gauge image unavailable", ln=True)
        return pdf.output(dest="S").encode("latin-1")

    pdf_bytes = build_pdf(inputs, prob, label)
    b64 = base64.b64encode(pdf_bytes).decode()
    st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="heart_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf">📥 Download PDF Report</a>', unsafe_allow_html=True)

# ---------------------------
# History preview
# ---------------------------
st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
st.markdown("<div class='card'>")
st.markdown("### Recent predictions")
if st.session_state["history"]:
    hist_df = pd.DataFrame([{ "time": h["ts"], "prob": f'{h["prob"]:.3f}', "label": h["label"] } for h in st.session_state["history"]])
    st.table(hist_df.head(8))
else:
    st.markdown("<div class='muted'>No predictions yet. Use the form to predict.</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div style='margin-top:10px; color:#6b7280; font-size:13px;'>Demo ML model — Not for clinical use. Validate clinically and legally before production.</div>", unsafe_allow_html=True)
