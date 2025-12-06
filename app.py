# app.py — CHD Predictor v5.0 (Premium HealthTech Redesign)
# Built by Sachin Ravi — v5.0 Premium
# - Uses best_heart_chd_model.joblib, scaler_chd.joblib, feature_order.json if present
# - If absent, will train from framingham.csv and export artifacts (recommended: run train_and_export.py first)
# - Modern two-pane layout, tabs, explainability, PDF export, history, clean feature alignment
# - SHAP primary, permutation fallback
# NOTE: This file is self-contained; copy to your project root alongside model artifacts.

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

# -----------------------
# Config & paths
# -----------------------
st.set_page_config(page_title="CHD Predictor — v5.0 Premium", layout="wide", initial_sidebar_state="collapsed")
ROOT = Path.cwd()
MODEL_PATH = ROOT / "best_heart_chd_model.joblib"
SCALER_PATH = ROOT / "scaler_chd.joblib"
FEATURE_ORDER_PATH = ROOT / "feature_order.json"
CSV_PATH = ROOT / "framingham.csv"
RANDOM_STATE = 42

# -----------------------
# Utility: loading dataset & artifacts
# -----------------------
def download_and_prepare_full_df():
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
    else:
        # fallback remote small copy (rare)
        try:
            df = pd.read_csv("https://raw.githubusercontent.com/ishank-j/propublica-tutorials/main/heart.csv")
        except Exception:
            # synthesize fallback (should not be used in production)
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
    # cleaning
    df = df.replace(["NA","?"," ", ""], np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(df.median(numeric_only=True))
    return df

def load_or_train_model():
    # If artifacts exist, load them and derive test split from dataset to be consistent
    if MODEL_PATH.exists() and SCALER_PATH.exists() and FEATURE_ORDER_PATH.exists():
        try:
            clf = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            with open(FEATURE_ORDER_PATH, "r") as f:
                features = json.load(f)
            # Build test-split for permutation importance later
            df = download_and_prepare_full_df()
            target_col = 'TenYearCHD' if 'TenYearCHD' in df.columns else df.columns[-1]
            X = df.drop(columns=[target_col])
            y = df[target_col].astype(int)
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.18, random_state=RANDOM_STATE, stratify=y)
            return clf, scaler, features, (X_test, y_test)
        except Exception:
            # fallback to retrain
            pass
    # Train a new model (safe)
    df = download_and_prepare_full_df()
    target_col = 'TenYearCHD' if 'TenYearCHD' in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight='balanced')
    clf.fit(X_train_s, y_train)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    feature_names = list(X.columns)
    with open(FEATURE_ORDER_PATH, "w") as f:
        json.dump(feature_names, f, indent=2)
    return clf, scaler, feature_names, (X_test, y_test)

model, scaler, FEATURE_NAMES, TEST_SPLIT = load_or_train_model()

# Ensure FEATURE_NAMES valid
if not FEATURE_NAMES or not isinstance(FEATURE_NAMES, list):
    # fallback derive
    if CSV_PATH.exists():
        df_tmp = pd.read_csv(CSV_PATH, nrows=1)
        cols = list(df_tmp.columns)
        if "TenYearCHD" in cols: cols.remove("TenYearCHD")
        FEATURE_NAMES = cols
    else:
        FEATURE_NAMES = ["age","male","education","currentSmoker","cigsPerDay","BPMeds","prevalentStroke","prevalentHyp","diabetes","totChol","sysBP","diaBP","BMI","heartRate","glucose"]

# -----------------------
# Feature alignment
# -----------------------
def align_features(df: pd.DataFrame, feature_names: list):
    df = df.copy()
    for c in feature_names:
        if c not in df.columns:
            df[c] = 0.0
    df = df[feature_names]
    return df.astype(float)

def preprocess_and_predict(df_aligned: pd.DataFrame):
    Xs = scaler.transform(df_aligned)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(Xs)[:,1][0])
    else:
        return float(model.predict(Xs)[0])

def risk_label(prob: float):
    if prob < 0.15: return "Low"
    if prob < 0.35: return "Moderate"
    if prob < 0.7: return "High"
    return "Very High"

def compute_model_accuracy():
    try:
        if TEST_SPLIT is not None:
            X_test, y_test = TEST_SPLIT
        else:
            df = download_and_prepare_full_df()
            target_col = 'TenYearCHD' if 'TenYearCHD' in df.columns else df.columns[-1]
            X = df.drop(columns=[target_col])
            y = df[target_col].astype(int)
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

# -----------------------
# UI styling (v5 premium)
# -----------------------
BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
:root{
  --bg-1: #071122;
  --bg-2: #0b1320;
  --card: rgba(255,255,255,0.03);
  --muted: #98a1b3;
  --accent: #00e0c8;
  --accent-2: #6c5ce7;
}
html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, "Helvetica Neue", Arial; background: linear-gradient(180deg,var(--bg-1),var(--bg-2)); color: #e6eef8; }
.topbar { display:flex; align-items:center; justify-content:space-between; padding:16px 8px; margin-bottom:8px; }
.brand { display:flex; gap:12px; align-items:center; }
.logo { width:44px; height:44px; border-radius:10px; background: linear-gradient(135deg, #06b6d4, #7c3aed); display:flex; align-items:center; justify-content:center; font-weight:800; color:white; box-shadow:0 8px 30px rgba(108,92,231,0.14); }
.h-title { font-weight:800; font-size:20px; }
.h-sub { font-size:12px; color:var(--muted); margin-top:2px; }
.card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:14px; padding:16px; border: 1px solid rgba(255,255,255,0.02); box-shadow: 0 10px 30px rgba(2,6,23,0.6); }
.left-panel { padding-right:12px; }
.form-grid { display:grid; grid-template-columns: repeat(3, 1fr); gap:12px; }
.form-row { margin-bottom:8px; }
.small-muted { color:var(--muted); font-size:12px; }
.section-title { font-weight:700; font-size:16px; margin-bottom:8px; }
.gauge-wrap { display:flex; align-items:center; justify-content:center; height:360px; }
.risk-badge { padding:8px 12px; border-radius:999px; font-weight:700; }
.pill-low { background: rgba(16,185,129,0.12); color:#10b981; border:1px solid rgba(16,185,129,0.12); }
.pill-moderate { background: rgba(245,158,11,0.08); color:#f59e0b; border:1px solid rgba(245,158,11,0.06); }
.pill-high { background: rgba(239,68,68,0.06); color:#ef4444; border:1px solid rgba(239,68,68,0.06); }
.badge { font-size:13px; }
.explain-row { display:flex; gap:12px; align-items:flex-start; }
.key-driver { background: rgba(255,255,255,0.02); padding:8px; border-radius:8px; min-width:120px; }
.footer-note { color:var(--muted); font-size:12px; margin-top:18px; }
@media (max-width:900px) {
  .form-grid { grid-template-columns: repeat(1, 1fr); }
  .gauge-wrap { height:260px; }
}
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)

# -----------------------
# Top bar
# -----------------------
left, right = st.columns([1,1])
with left:
    st.markdown(f"""
    <div class="topbar">
      <div class="brand">
        <div class="logo">❤️</div>
        <div>
          <div class="h-title">Clinical Heart Risk — v5.0</div>
          <div class="h-sub">Built by Sachin Ravi • Premium HealthTech demo</div>
        </div>
      </div>
      <div style="display:flex; gap:10px; align-items:center;">
        <div class="small-muted">Model accuracy: <strong style="color:#e6eef8">{ACC_STR}</strong></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------
# Tabs (Predict / Explain / History / Report / About)
# -----------------------
tab = st.tabs(["Predict", "Explain", "History", "Report", "About"])[0]  # placeholder; we'll use st.tabs below

tabs = st.tabs(["Predict","Explain","History","Report","About"])
tab_predict, tab_explain, tab_history, tab_report, tab_about = tabs

# -------------
# Predict tab
# -------------
with tab_predict:
    col_left, col_right = st.columns([1.05,0.9], gap="large")
    with col_left:
        st.markdown('<div class="card left-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Patient details</div>', unsafe_allow_html=True)

        # Multi-column form grid
        st.markdown('<div class="form-grid">', unsafe_allow_html=True)
        # Widgets - ensure keys are stable and mapable
        age = st.number_input("Age", min_value=18, max_value=120, value=58, key="v5_age")
        gender = st.selectbox("Gender", options=["Male","Female"], index=0, key="v5_gender")
        education = st.selectbox("Education (1-4)", options=[1,2,3,4], index=1, key="v5_education")
        smoker = st.selectbox("Current smoker?", options=["Smoker","Non-smoker"], index=0, key="v5_smoker")
        cigsPerDay = st.number_input("Cigarettes / day", min_value=0, max_value=100, value=20, key="v5_cigs")
        BPMeds = st.selectbox("On BP medication?", options=["Yes","No"], index=0, key="v5_bpmeds")
        prevalentStroke = st.selectbox("Stroke history?", options=["Yes","No"], index=0, key="v5_stroke")
        prevalentHyp = st.selectbox("Hypertension?", options=["Yes","No"], index=0, key="v5_hyp")
        diabetes = st.selectbox("Diabetes?", options=["Yes","No"], index=0, key="v5_diab")
        totChol = st.number_input("Total cholesterol (mg/dL)", min_value=100.0, max_value=600.0, value=250.0, key="v5_chol")
        sysBP = st.number_input("Systolic BP (mmHg)", min_value=80.0, max_value=260.0, value=120.0, key="v5_sys")
        diaBP = st.number_input("Diastolic BP (mmHg)", min_value=40.0, max_value=160.0, value=80.0, key="v5_dia")
        BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, key="v5_bmi")
        heartRate = st.number_input("Heart Rate (bpm)", min_value=30.0, max_value=200.0, value=72.0, key="v5_hr")
        glucose = st.number_input("Glucose (mg/dL)", min_value=40.0, max_value=400.0, value=95.0, key="v5_gl")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
        submit = st.button("Predict Risk", key="v5_predict", help="Run prediction using the trained CHD model", use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Risk Summary</div>', unsafe_allow_html=True)
        # place for gauge + badge + recommendations
        gauge_placeholder = st.empty()
        badge_placeholder = st.empty()
        rec_placeholder = st.empty()
        explain_preview = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    # session history initialization
    if "history_v5" not in st.session_state:
        st.session_state["history_v5"] = []

    # Helper: build input dict mapping to FEATURE_NAMES order
    def build_input_ordered():
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
        ordered = {fn: mapping.get(fn, 0.0) for fn in FEATURE_NAMES}
        return ordered

    # Prediction logic
    if submit:
        input_ordered = build_input_ordered()
        X_user = pd.DataFrame([input_ordered])
        X_aligned = align_features(X_user, FEATURE_NAMES)
        try:
            prob = preprocess_and_predict(X_aligned)
            label = risk_label(prob)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            prob = None
            label = "N/A"
        # save history
        rec = {"ts": datetime.now().isoformat(timespec='seconds'), "inputs": input_ordered, "prob": prob, "label": label}
        st.session_state["history_v5"].insert(0, rec)
        st.session_state["history_v5"] = st.session_state["history_v5"][:60]

        # Render gauge + badge + rec
        if prob is not None:
            # gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob*100,
                number={'suffix':'%','font':{'size':44,'color':'#e6eef8'}},
                title={'text': "10-year CHD risk", 'font': {'size':14}},
                gauge={
                    'axis': {'range':[0,100]},
                    'bar': {'color':'#00e0c8'},
                    'steps': [
                        {'range':[0,15],'color':'#10b981'},
                        {'range':[15,35],'color':'#f59e0b'},
                        {'range':[35,70],'color':'#fb923c'},
                        {'range':[70,100],'color':'#ef4444'}
                    ],
                    'threshold': {'line': {'color':'#ffd166','width':4}, 'thickness':0.8, 'value': prob*100}
                }
            ))
            fig.update_layout(margin=dict(t=30,b=10,l=20,r=20), paper_bgcolor='rgba(0,0,0,0)', height=300)
            gauge_placeholder.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
            # badge
            color_class = "pill-low" if label=="Low" else ("pill-moderate" if label=="Moderate" else ("pill-high" if label=="High" else "pill-high"))
            badge_html = f'<div style="display:flex; gap:10px; align-items:center;"><div class="risk-badge {color_class} badge">{label}</div><div class="small-muted" style="margin-left:8px">Probability: <strong>{prob:.3f}</strong></div></div>'
            badge_placeholder.markdown(badge_html, unsafe_allow_html=True)
            # recommendations
            if prob > 0.7:
                rec_placeholder.markdown("<div style='font-weight:700'>Urgent clinical review recommended</div><ul><li>ECG & cardiology referral</li><li>Fasting glucose & lipid panel</li><li>Consider immediate evaluation</li></ul>", unsafe_allow_html=True)
            elif prob > 0.35:
                rec_placeholder.markdown("<div style='font-weight:700'>High risk — consider follow-up</div><ul><li>BP control</li><li>Lifestyle modification</li><li>Schedule clinical tests</li></ul>", unsafe_allow_html=True)
            else:
                rec_placeholder.markdown("<div style='font-weight:700'>Low risk — maintain screening</div><ul><li>Healthy diet & activity</li><li>Annual checkups</li></ul>", unsafe_allow_html=True)

# -----------------------
# Explain tab
# -----------------------
with tab_explain:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">SHAP explanation will be attempted first. If unavailable, permutation importance is displayed (aligned to training features).</div>', unsafe_allow_html=True)

    use_perm = False
    shap_html = st.empty()
    pi_html = st.empty()

    # Prepare a test batch for permutation importance / SHAP
    try:
        X_test_full, y_test_full = TEST_SPLIT
        X_test_aligned = align_features(X_test_full, FEATURE_NAMES)
    except Exception:
        df_all = download_and_prepare_full_df()
        target_col = 'TenYearCHD' if 'TenYearCHD' in df_all.columns else df_all.columns[-1]
        X_tmp = df_all.drop(columns=[target_col])
        X_test_aligned = align_features(X_tmp, FEATURE_NAMES)

    # Try SHAP
    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            # use small sample for performance
            sample = X_test_aligned.sample(min(120, len(X_test_aligned)), random_state=RANDOM_STATE)
            shap_vals = explainer.shap_values(sample)
            # shap_vals could be list for binary
            if isinstance(shap_vals, list):
                shap_mean = np.mean(np.abs(shap_vals[1]), axis=0)
            else:
                shap_mean = np.mean(np.abs(shap_vals), axis=0)
            df_shap = pd.DataFrame({"feature": FEATURE_NAMES, "importance": shap_mean}).sort_values("importance", ascending=True)
            fig_shap = px.bar(df_shap, x="importance", y="feature", orientation="h", height=420)
            fig_shap.update_layout(margin=dict(t=10,b=10,l=120,r=10), paper_bgcolor='rgba(0,0,0,0)')
            shap_html.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar":True})
        except Exception as e:
            st.warning("Advanced SHAP explanation unavailable — switching to permutation importance.")
            use_perm = True
    else:
        st.info("SHAP package not installed — showing permutation importance.")
        use_perm = True

    if use_perm:
        try:
            from sklearn.inspection import permutation_importance
            # sample subset for speed
            Xb = X_test_aligned
            yb = y_test_full if 'y_test_full' in locals() else None
            if yb is None:
                # fallback: build from full df
                df_all = download_and_prepare_full_df()
                target_col = 'TenYearCHD' if 'TenYearCHD' in df_all.columns else df_all.columns[-1]
                yb = df_all[target_col].astype(int).values
            res = permutation_importance(model, Xb, yb, n_repeats=8, random_state=RANDOM_STATE, n_jobs=1)
            pi = res.importances_mean
            df_pi = pd.DataFrame({"feature": FEATURE_NAMES, "importance": pi}).sort_values("importance", ascending=True)
            fig_pi = px.bar(df_pi, x="importance", y="feature", orientation="h", height=420)
            fig_pi.update_layout(margin=dict(t=10,b=10,l=120,r=10), paper_bgcolor='rgba(0,0,0,0)')
            pi_html.plotly_chart(fig_pi, use_container_width=True, config={"displayModeBar":True})
        except Exception as e:
            st.error(f"Permutation importance failed: {e}")
            st.info("Explainability is unavailable for this model.")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# History tab
# -----------------------
with tab_history:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Recent predictions</div>', unsafe_allow_html=True)
    hist = st.session_state.get("history_v5", [])
    if hist:
        df_hist = pd.DataFrame([{"time": h["ts"], "prob": f'{h["prob"]:.3f}', "label": h["label"]} for h in hist])
        st.dataframe(df_hist, height=260)
        # download CSV
        csv = df_hist.to_csv(index=False).encode('utf-8')
        b64 = base64.b64encode(csv).decode()
        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="predictions_history.csv">Download history CSV</a>', unsafe_allow_html=True)
    else:
        st.markdown("<div class='small-muted'>No predictions yet — use the Predict tab.</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# Report tab (PDF export)
# -----------------------
with tab_report:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Export / Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">Create a polished patient PDF report (includes gauge snapshot and top drivers).</div>', unsafe_allow_html=True)

    if st.button("Generate PDF from last prediction"):
        if not st.session_state.get("history_v5"):
            st.warning("No prediction available — run a prediction first.")
        else:
            last = st.session_state["history_v5"][0]
            inputs = last["inputs"]
            prob = last["prob"]
            label = last["label"]

            # Build PDF
            pdf = FPDF()
            pdf.set_auto_page_break(True, margin=12)
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 8, "CHD Predictor Report — v5.0", ln=True, align="C")
            pdf.set_font("Arial", size=10)
            pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            pdf.ln(6)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 6, "Inputs", ln=True)
            pdf.set_font("Arial", size=10)
            for k,v in inputs.items():
                pdf.cell(0, 6, f"{k}: {v}", ln=True)
            pdf.ln(6)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 6, "Prediction", ln=True)
            pdf.set_font("Arial", size=10)
            pdf.cell(0, 6, f"Probability: {prob:.3f}", ln=True)
            pdf.cell(0, 6, f"Risk level: {label}", ln=True)
            # include simple bar chart of top importance (permutation if available)
            try:
                # try permutation importances
                from sklearn.inspection import permutation_importance
                X_test_full, y_test_full = TEST_SPLIT
                X_test_aligned = align_features(X_test_full, FEATURE_NAMES)
                res = permutation_importance(model, X_test_aligned.sample(min(120,len(X_test_aligned)), random_state=RANDOM_STATE), y_test_full.sample(min(120,len(y_test_full)), random_state=RANDOM_STATE), n_repeats=6, random_state=RANDOM_STATE)
                pi = res.importances_mean
                df_pi = pd.DataFrame({"feature": FEATURE_NAMES, "importance": pi}).sort_values("importance", ascending=False).head(6)
                fig = px.bar(df_pi, x="importance", y="feature", orientation="h", height=220)
                img_bytes = fig.to_image(format="png", scale=2)
                tmp = "tmp_pi.png"
                with open(tmp, "wb") as f:
                    f.write(img_bytes)
                pdf.image(tmp, w=170)
            except Exception:
                pdf.cell(0, 6, "Feature importance image unavailable.", ln=True)
            pdf_bytes = pdf.output(dest="S").encode("latin-1")
            b64 = base64.b64encode(pdf_bytes).decode()
            st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="chd_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf">📥 Download PDF Report</a>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# About tab
# -----------------------
with tab_about:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">About this model</div>', unsafe_allow_html=True)
    st.markdown("""
    **CHD Predictor v5.0** — Premium HealthTech demo built by **Sachin Ravi**.

    **Model**: RandomForestClassifier (trained on Framingham-like dataset).  
    **Artifacts**: `best_heart_chd_model.joblib`, `scaler_chd.joblib`, `feature_order.json`.  
    **Explainability**: SHAP (TreeExplainer) attempted, permutation importance fallback.  
    **Notes**: This is a demo ML model for educational use. Not for clinical decision-making.

    **Recommended production steps**:
    - Clinical validation & regulatory review
    - Secure user authentication & audit logs
    - Data encryption & privacy safeguards
    - Continuous monitoring & model performance checks
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# Footer / small notes
# -----------------------
st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
st.markdown('<div class="footer-note">Demo ML model — Not for clinical use. Validate clinically & legally before production.</div>', unsafe_allow_html=True)
