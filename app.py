# app.py — Final CHD Predictor (v2.1) — Built by Sachin Ravi
# End-to-end dashboard: model load/train, accuracy metric, SHAP + permutation fallback, PDF export, premium UI.

import os
import io
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

# Optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="CHD Predictor — Built by Sachin Ravi", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# Styling & small helpers
# ---------------------------
def inject_css():
    css = r"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    :root{
      --bg:#0b0f12;
      --card:#0f1316;
      --muted: #9aa3ab;
      --accent: #00E1C5;
    }

    html, body, [class*="css"]  {
      font-family: Inter, system-ui, -apple-system, "SF Pro Text", "Helvetica Neue", Arial;
      background: linear-gradient(180deg, rgba(6,8,11,1) 0%, rgba(8,10,13,1) 100%) fixed;
      color: #e6eef3;
    }

    .bg-gradient {
      position: absolute;
      inset: 0;
      z-index: -1;
      background: radial-gradient(800px 400px at 10% 10%, rgba(0,230,197,0.04), transparent 8%),
                  radial-gradient(600px 300px at 90% 80%, rgba(95,75,255,0.02), transparent 6%);
      pointer-events: none;
    }

    .glass {
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.03);
      padding: 14px;
      box-shadow: 0 10px 26px rgba(2,6,23,0.5);
    }

    .hero-title { font-size:24px; font-weight:700; letter-spacing:0.2px; }
    .hero-sub { color: var(--muted); margin-top:6px; font-size:13px; }

    .stButton>button { border-radius:10px !important; padding:8px 14px !important; }
    .chip { display:inline-block; padding:6px 10px; border-radius:999px; background: rgba(255,255,255,0.03); color:#d9fef0; font-size:13px; margin-right:6px; border:1px solid rgba(255,255,255,0.02); }
    .soft-divider { height:1px; background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); margin:12px 0; border-radius:2px; }
    .muted { color:var(--muted); font-size:13px; }
    .lottie-wrap { width:64px; height:64px; display:inline-block; }
    .stDataFrame table { border-collapse:separate !important; border-spacing:0 6px; }
    </style>
    <div class="bg-gradient"></div>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_css()

# ---------------------------
# Config / model file paths
# ---------------------------
MODEL_PATH = "best_heart_chd_model.joblib"
SCALER_PATH = "scaler_chd.joblib"
RANDOM_STATE = 42

# ---------------------------
# Dataset utilities (download or synth)
# ---------------------------
def download_dataset_or_synthesize():
    """Try to download a public heart dataset; otherwise synthesize one for demo."""
    url = "https://raw.githubusercontent.com/ishank-j/propublica-tutorials/main/heart.csv"
    try:
        df = pd.read_csv(url)
    except Exception:
        np.random.seed(RANDOM_STATE)
        n = 1000
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
    return df

# ---------------------------
# Model load or train
# ---------------------------
def train_and_save_default_model():
    df = download_dataset_or_synthesize()
    # pick target column if available
    target_col = 'TenYearCHD' if 'TenYearCHD' in df.columns else ( 'target' if 'target' in df.columns else df.columns[-1] )
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    clf = RandomForestClassifier(n_estimators=250, random_state=RANDOM_STATE)
    clf.fit(X_train_s, y_train)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return clf, scaler, list(X.columns), (X_test, y_test)

def load_model_and_scaler():
    # returns model, scaler, feature_names, test_split(if available)
    if Path(MODEL_PATH).exists() and Path(SCALER_PATH).exists():
        try:
            clf = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            # attempt to build a test split for accuracy calculation by downloading dataset
            try:
                df = download_dataset_or_synthesize()
                target_col = 'TenYearCHD' if 'TenYearCHD' in df.columns else ( 'target' if 'target' in df.columns else df.columns[-1] )
                X = df.drop(columns=[target_col])
                y = df[target_col].astype(int)
                _, X_test, _, y_test = train_test_split(X, y, test_size=0.18, random_state=RANDOM_STATE, stratify=y)
                return clf, scaler, list(X.columns), (X_test, y_test)
            except Exception:
                return clf, scaler, None, None
        except Exception:
            pass
    # fallback: train
    clf, scaler, features, test_split = train_and_save_default_model()
    return clf, scaler, features, test_split

model, scaler, FEATURE_NAMES, TEST_SPLIT = load_model_and_scaler()

# ---------------------------
# Preprocess & predict helpers
# ---------------------------
def preprocess_df_for_model(df_in: pd.DataFrame):
    df = df_in.copy().astype(float)
    try:
        Xs = scaler.transform(df)
    except Exception:
        tmp = StandardScaler()
        Xs = tmp.fit_transform(df)
    return Xs

def predict_prob_single(df_in: pd.DataFrame):
    Xs = preprocess_df_for_model(df_in)
    try:
        proba = model.predict_proba(Xs)[:,1]
    except Exception:
        proba = model.predict(Xs).astype(float)
    return float(proba[0])

def risk_label(prob):
    if prob < 0.15:
        return "Low"
    if prob < 0.35:
        return "Moderate"
    if prob < 0.7:
        return "High"
    return "Very High"

# ---------------------------
# Model accuracy calculation (best-effort)
# ---------------------------
def compute_model_accuracy():
    """
    Try to compute model accuracy using TEST_SPLIT if available.
    If not available, attempt to download dataset and evaluate.
    If impossible, return None.
    """
    try:
        if TEST_SPLIT is not None:
            X_test, y_test = TEST_SPLIT
        else:
            df = download_dataset_or_synthesize()
            target_col = 'TenYearCHD' if 'TenYearCHD' in df.columns else ( 'target' if 'target' in df.columns else df.columns[-1] )
            X = df.drop(columns=[target_col])
            y = df[target_col].astype(int)
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.18, random_state=RANDOM_STATE, stratify=y)
        Xs = scaler.transform(X_test)
        try:
            ypred = model.predict(Xs)
        except Exception:
            # fallback: threshold on predict_proba if available
            if hasattr(model, "predict_proba"):
                ypred = (model.predict_proba(Xs)[:,1] >= 0.5).astype(int)
            else:
                return None
        acc = accuracy_score(y_test, ypred)
        return float(acc)
    except Exception:
        return None

MODEL_ACCURACY = compute_model_accuracy()

# ---------------------------
# Plot helpers
# ---------------------------
def create_dual_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        title={'text': "10-year CHD (%)"},
        gauge={
            'axis': {'range': [0,100]},
            'bar': {'color': "#ff6b6b"},
            'steps': [
                {'range':[0,15],'color':'#2ECC71'},
                {'range':[15,35],'color':'#F1C40F'},
                {'range':[35,70],'color':'#E67E22'},
                {'range':[70,100],'color':'#E74C3C'}
            ],
            'threshold': {'line': {'color':'white','width':4}, 'thickness':0.75, 'value': prob*100}
        }
    ))
    fig.update_layout(height=360, margin=dict(t=15,b=15,l=15,r=15), paper_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_feature_chips(contribs):
    chips_html = "<div style='display:flex; gap:6px; flex-wrap:wrap;'>"
    for f, v in contribs.items():
        color = "#ff6b6b" if v>0 else "#37c77f"
        sign = "+" if v>0 else ""
        chips_html += f"<div class='chip' style='border:1px solid rgba(255,255,255,0.02)'><strong>{f}</strong>: {sign}{v:.2f}</div>"
    chips_html += "</div>"
    return chips_html

# ---------------------------
# UI: Header, Sidebar
# ---------------------------
# Header (minimal clean)
st.markdown("""
<div style="display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:12px;">
  <div style="display:flex; gap:12px; align-items:center;">
    <div style="width:48px; height:48px; border-radius:10px; background:linear-gradient(135deg,#00E1C5,#6EE7B7); display:flex; align-items:center; justify-content:center; font-weight:700; color:#02121b;">❤️</div>
    <div>
      <div class="hero-title">CHD Predictor</div>
      <div class="hero-sub">Built by Sachin Ravi</div>
    </div>
  </div>
  <div style="display:flex; gap:10px; align-items:center;">
    <div class="muted">Model accuracy: <strong>{:.2%}</strong></div>
  </div>
</div>
""".format(MODEL_ACCURACY if MODEL_ACCURACY is not None else 0.0), unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("## CHD Predictor")
    st.markdown("Built by Sachin Ravi")
    st.markdown("---")
    st.markdown("### Presets")
    preset = st.selectbox("Load preset", options=["Custom","Healthy (demo)","High-risk smoker","Elderly hypertensive"])
    st.button("Clear preset", key="clear_preset")
    st.markdown("---")
    st.markdown("### Notes")
    st.markdown("<div class='muted'>This is a demo ML model. Validate before clinical use.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Input form (3-column)
# ---------------------------
left_col, mid_col, right_col = st.columns([1.1,1.1,1.0], gap="large")

# defaults
defaults = {
    "age": 58,
    "male": 1,
    "education": 1,
    "currentSmoker": 1,
    "cigsPerDay": 20,
    "BPMeds": 1,
    "prevalentStroke": 0,
    "prevalentHyp": 1,
    "diabetes": 1,
    "totChol": 250,
    "sysBP": 160,
    "diaBP": 95,
    "BMI": 29.5,
    "heartRate": 90,
    "glucose": 140
}

with st.form("input_form"):
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### Patient details — enter or load a preset")

    with left_col:
        age = st.number_input("Age", min_value=0, max_value=120, value=int(defaults["age"]), key="age")
        gender = st.selectbox("Gender", options=[("Male",1),("Female",0)], format_func=lambda x:x[0], index=0)
        education = st.selectbox("Education (1-4)", options=[1,2,3,4], index=int(defaults["education"])-1)
        currentSmoker = st.selectbox("Current smoker?", options=[("Yes",1),("No",0)], format_func=lambda x:x[0], index=0 if int(defaults["currentSmoker"]) else 1)
        cigsPerDay = st.number_input("Cigarettes / day", min_value=0, max_value=100, value=int(defaults["cigsPerDay"]))

    with mid_col:
        BPMeds = st.selectbox("On BP medication?", options=[("Yes",1),("No",0)], format_func=lambda x:x[0], index=0 if int(defaults["BPMeds"]) else 1)
        prevalentStroke = st.selectbox("Stroke history?", options=[("Yes",1),("No",0)], format_func=lambda x:x[0], index=1 if int(defaults["prevalentStroke"])==0 else 0)
        prevalentHyp = st.selectbox("Hypertension?", options=[("Yes",1),("No",0)], format_func=lambda x:x[0], index=0 if int(defaults["prevalentHyp"]) else 1)
        diabetes = st.selectbox("Diabetes?", options=[("Yes",1),("No",0)], format_func=lambda x:x[0], index=0 if int(defaults["diabetes"]) else 1)

    with right_col:
        totChol = st.number_input("Total cholesterol", min_value=100.0, max_value=600.0, value=float(defaults["totChol"]))
        sysBP = st.number_input("Systolic BP", min_value=60.0, max_value=240.0, value=float(defaults["sysBP"]))
        diaBP = st.number_input("Diastolic BP", min_value=30.0, max_value=140.0, value=float(defaults["diaBP"]))
        heartRate = st.number_input("Heart Rate", min_value=30.0, max_value=200.0, value=float(defaults["heartRate"]))
        BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=float(defaults["BMI"]))
        glucose = st.number_input("Glucose", min_value=40.0, max_value=400.0, value=float(defaults["glucose"]))

    st.markdown("</div>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1,1,1])
    with col_a:
        predict_btn = st.form_submit_button("🔍 Predict Risk")
    with col_b:
        compare_mode = st.checkbox("Enable Before / After comparison", value=False)
    with col_c:
        save_session = st.checkbox("Save to session history", value=True)

# apply presets
if preset != "Custom" and not predict_btn:
    if preset == "Healthy (demo)":
        st.session_state.update({"age":40,"currentSmoker":("No",0),"cigsPerDay":0,"totChol":170,"sysBP":120,"diaBP":76,"BMI":22,"glucose":90,"heartRate":72})
    elif preset == "High-risk smoker":
        st.session_state.update({"age":62,"currentSmoker":("Yes",1),"cigsPerDay":25,"totChol":270,"sysBP":155,"diaBP":96,"BMI":31,"glucose":145,"heartRate":92})
    elif preset == "Elderly hypertensive":
        st.session_state.update({"age":73,"currentSmoker":("No",0),"cigsPerDay":0,"totChol":260,"sysBP":170,"diaBP":98,"BMI":28,"glucose":130,"heartRate":86})
    st.experimental_rerun()

# session history
if 'history' not in st.session_state:
    st.session_state['history'] = []

def build_input_dict():
    gval = gender[1] if isinstance(gender, tuple) else gender
    cur_sm = currentSmoker[1] if isinstance(currentSmoker, tuple) else currentSmoker
    data = {
        "age": float(age),
        "male": float(gval),
        "education": float(education),
        "currentSmoker": float(cur_sm),
        "cigsPerDay": float(cigsPerDay),
        "BPMeds": float(BPMeds[1]) if isinstance(BPMeds, tuple) else float(BPMeds),
        "prevalentStroke": float(prevalentStroke[1]) if isinstance(prevalentStroke, tuple) else float(prevalentStroke),
        "prevalentHyp": float(prevalentHyp[1]) if isinstance(prevalentHyp, tuple) else float(prevalentHyp),
        "diabetes": float(diabetes[1]) if isinstance(diabetes, tuple) else float(diabetes),
        "totChol": float(totChol),
        "sysBP": float(sysBP),
        "diaBP": float(diaBP),
        "BMI": float(BMI),
        "heartRate": float(heartRate),
        "glucose": float(glucose)
    }
    return data

# ---------------------------
# Prediction action
# ---------------------------
if predict_btn:
    inputs = build_input_dict()
    X_df = pd.DataFrame([inputs])

    # align columns if FEATURE_NAMES exist
    try:
        if FEATURE_NAMES and len(FEATURE_NAMES) == X_df.shape[1]:
            X_df = X_df[FEATURE_NAMES]
    except Exception:
        pass

    # loader
    with st.container():
        st.markdown('<div class="glass" style="padding:12px;">', unsafe_allow_html=True)
        st.markdown("<div style='display:flex; gap:12px; align-items:center;'>", unsafe_allow_html=True)
        st.markdown("<div class='lottie-wrap' id='lottie'></div>", unsafe_allow_html=True)
        st.markdown("<div><strong>Predicting risk...</strong><div class='muted'>Running model & explainability</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    time.sleep(0.6)

    try:
        prob = predict_prob_single(X_df)
        label = risk_label(prob)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # save session entry
    hist_item = {"timestamp": datetime.now().isoformat(timespec='seconds'), "inputs": inputs, "prob": prob, "label": label}
    if save_session:
        st.session_state['history'].insert(0, hist_item)
        st.session_state['history'] = st.session_state['history'][:20]

    # Top cards (show model accuracy if available)
    col1, col2, col3 = st.columns([1,2,1], gap="large")
    with col1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:13px; color:var(--muted)'>Risk</div><h3 style='margin:6px 0; color:{'#ff6b6b' if prob>0.7 else '#f1c40f' if prob>0.35 else '#37c77f'}'>{label}</h3>", unsafe_allow_html=True)
        st.markdown(f"<div class='muted'>Probability: <strong>{prob:.2f}</strong></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("<div style='font-size:13px; color:var(--muted)'>Recommendation</div>", unsafe_allow_html=True)
        if prob > 0.7:
            st.markdown("<div style='font-weight:700; margin-top:6px;'>Urgent clinical review recommended</div>", unsafe_allow_html=True)
            st.write("Recommend: ECG, fasting glucose, lipid panel, cardiology consult.")
        elif prob > 0.35:
            st.markdown("<div style='font-weight:700; margin-top:6px;'>High risk — consider follow-up</div>", unsafe_allow_html=True)
            st.write("Lifestyle changes, BP control, consider further testing.")
        else:
            st.markdown("<div style='font-weight:700; margin-top:6px;'>Low risk — maintain screening</div>", unsafe_allow_html=True)
            st.write("Lifestyle, routine checks.")
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("<div style='font-size:13px; color:var(--muted)'>Model Accuracy</div>", unsafe_allow_html=True)
        if MODEL_ACCURACY is not None:
            st.markdown(f"<h3 style='margin:6px 0;'>{MODEL_ACCURACY:.2%}</h3>", unsafe_allow_html=True)
            st.markdown("<div class='muted'>Calculated on test split / dataset</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='muted'>Accuracy unavailable</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)

    # Visuals: gauge + explainability
    left_viz, right_viz = st.columns([2,1], gap="large")
    with left_viz:
        fig_g = create_dual_gauge(prob)
        st.plotly_chart(fig_g, use_container_width=True)
        st.markdown(f"<div style='text-align:center; font-size:20px; margin-top:-10px;'>{prob*100:.1f}% — {label} risk</div>", unsafe_allow_html=True)
    with right_viz:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("<div style='font-size:14px; font-weight:700;'>Explainability</div>", unsafe_allow_html=True)

        feat_names = list(pd.DataFrame([inputs]).columns)
        X_scaled = preprocess_df_for_model(pd.DataFrame([inputs]))

        use_permutation = False

        # SHAP primary
        if SHAP_AVAILABLE:
            try:
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_scaled)
                if isinstance(shap_vals, list):
                    shap_single = shap_vals[1][0]
                else:
                    shap_single = shap_vals[0]
                contribs = {feat_names[i]: float(shap_single[i]) for i in range(len(feat_names))}
                top6 = dict(sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:6])
                st.markdown(plot_feature_chips(top6), unsafe_allow_html=True)
                df_bar = pd.DataFrame({"feature": list(top6.keys()), "impact": [top6[k] for k in top6]})
                st.plotly_chart(px.bar(df_bar, x="impact", y="feature", orientation="h", height=260), use_container_width=True)
            except Exception:
                st.warning("SHAP failed — switching to permutation importance")
                use_permutation = True
        else:
            st.warning("SHAP not installed — using permutation importance fallback")
            use_permutation = True

        # Permutation importance fallback
        if use_permutation:
            try:
                from sklearn.inspection import permutation_importance
                result = permutation_importance(model, X_scaled, model.predict(X_scaled), n_repeats=15, random_state=42)
                pi_vals = result.importances_mean
                df_pi = pd.DataFrame({"feature": feat_names, "importance": pi_vals}).sort_values("importance", ascending=False)
                st.plotly_chart(px.bar(df_pi, x="importance", y="feature", orientation="h", height=260), use_container_width=True)
            except Exception as e:
                st.error(f"Permutation importance failed: {e}")
                st.info("Explainability unavailable for this model.")

        st.markdown("</div>", unsafe_allow_html=True)

    # Comparison mode
    if compare_mode:
        st.markdown("<div style='margin-top:12px;' class='glass'>", unsafe_allow_html=True)
        st.markdown("### Before → After slider (simulate interventions)")
        sim_col1, sim_col2 = st.columns([1,2])
        with sim_col1:
            reduce_bp = st.slider("Reduce systolic BP by (mmHg)", 0, 60, 10)
            reduce_chol = st.slider("Reduce cholesterol by (mg/dL)", 0, 100, 20)
            stop_smoking = st.checkbox("Stop smoking (simulate)", value=False)
        with sim_col2:
            after_inputs = inputs.copy()
            after_inputs["sysBP"] = max(80, after_inputs["sysBP"] - reduce_bp)
            after_inputs["totChol"] = max(100, after_inputs["totChol"] - reduce_chol)
            if stop_smoking:
                after_inputs["currentSmoker"] = 0
                after_inputs["cigsPerDay"] = 0
            prob_after = predict_prob_single(pd.DataFrame([after_inputs]))
            cols_v = st.columns(2)
            cols_v[0].metric("Before risk", f"{prob*100:.1f}%")
            cols_v[1].metric("After risk", f"{prob_after*100:.1f}%")
            delta = (prob - prob_after) * 100
            st.markdown(f"**Estimated improvement:** {delta:.2f} percentage points", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Session history / preview
# ---------------------------
st.markdown("<div style='margin-top:14px;' class='glass'>", unsafe_allow_html=True)
st.markdown("### Model input preview")
if st.session_state['history']:
    last = st.session_state['history'][0]
    preview_df = pd.DataFrame(list(last['inputs'].items()), columns=["feature","value"])
    st.dataframe(preview_df, use_container_width=True, height=200)
else:
    st.markdown("<div class='muted'>No saved sessions — make a prediction and check 'Save to session history'.</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Session history (last 20)"):
    for idx, h in enumerate(st.session_state['history']):
        cols = st.columns([2,1,1])
        cols[0].markdown(f"**{h['timestamp']}** — {h['label']} — {h['prob']:.2f}")
        if cols[1].button("Load", key=f"load_{idx}"):
            for k,v in h['inputs'].items():
                # set simple session states (best-effort)
                try:
                    st.session_state[k] = v
                except Exception:
                    pass
            st.experimental_rerun()
        if cols[2].button("Export PDF", key=f"pdf_{idx}"):
            pdf_bytes = build_pdf(bytes_flag=True, inputs=h['inputs'], prob=h['prob'], label=h['label'])
            b64 = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="heart_report_{h["timestamp"].replace(":","")}.pdf">Download</a>'
            st.markdown(href, unsafe_allow_html=True)

# ---------------------------
# PDF utils
# ---------------------------
def fig_to_bytes(fig, fmt="png"):
    try:
        return fig.to_image(format=fmt, scale=2)
    except Exception:
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        return buf.read()

def build_pdf(bytes_flag=True, inputs=None, prob=0.0, label=""):
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
    if inputs:
        for k,v in inputs.items():
            pdf.cell(0, 6, f"{k}: {v}", ln=True)
    pdf.ln(6)
    pdf.set_font("Arial", size=12, style="B")
    pdf.cell(0, 6, "Prediction", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 6, f"Probability: {prob:.3f}", ln=True)
    pdf.cell(0, 6, f"Risk level: {label}", ln=True)
    try:
        fig = create_dual_gauge(prob)
        img = fig_to_bytes(fig)
        tmp = "tmp_gauge.png"
        with open(tmp, "wb") as f:
            f.write(img)
        pdf.image(tmp, w=160)
    except Exception:
        pdf.cell(0, 6, "Gauge image unavailable", ln=True)
    out = pdf.output(dest="S").encode("latin-1")
    return out

# ---------------------------
# Lottie + keyboard shortcuts
# ---------------------------
LOTTIE_HEART = "https://assets8.lottiefiles.com/packages/lf20_j1adxtyb.json"
lottie_js = f"""
<script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
<lottie-player src="{LOTTIE_HEART}"  background="transparent"  speed="1"  style="width:64px; height:64px;"  loop  autoplay></lottie-player>

<script>
document.addEventListener('keydown', function(e) {{
  if (e.key === 'r' || e.key === 'R') {{
    const btn = [...document.querySelectorAll('button')].find(b => b.innerText && b.innerText.includes('Predict'));
    if(btn) btn.click();
  }}
  if (e.key === 'p' || e.key === 'P') {{
    const pdfBtn = [...document.querySelectorAll('button')].find(b => b.innerText && (b.innerText.includes('Download PDF') || b.innerText.includes('Download report')));
    if(pdfBtn) pdfBtn.click();
  }}
}});
</script>
"""
st_html(lottie_js, height=95)

# Footer note
st.markdown("<div style='margin-top:10px; color:var(--muted); font-size:13px;'>Demo ML model. Not for clinical use. Validate dataset, model, and regulatory needs before production.</div>", unsafe_allow_html=True)
