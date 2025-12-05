# app.py — CHD Predictor v3.0 (Modern HealthTech Pro)
# Built by Sachin Ravi
# Matches training pipeline from framingham.csv (TenYearCHD target)
# - Feature alignment from local framingham.csv (guarantees ordering)
# - Loads best_heart_chd_model.joblib and scaler_chd.joblib if present
# - Trains & saves model if not present (same logic as your training code)
# - SHAP + permutation importance fallback (on aligned test batch)
# - Clean, stable UI; no session_state input corruption; no background flicker

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

# optional shap
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="CHD Predictor — Built by Sachin Ravi (v3.0)",
                   layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# Lightweight CSS (stable background, no flicker)
# ---------------------------
def inject_css(bg_enabled: bool, bg_url: str):
    # ensure background is static and lightweight
    bg_css = f"background-image: url('{bg_url}'); background-size: cover; background-position: center; filter: blur(6px) saturate(0.75);" if bg_enabled else ""
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    html, body, [class*="css"] {{
        font-family: Inter, system-ui, -apple-system, "SF Pro Text", "Helvetica Neue", Arial;
        color: #0f1724;
        background: #f7fafc;
    }}
    .bg-static {{
        position: fixed;
        inset: 0;
        z-index: -2;
        {bg_css}
        opacity: 0.16;
        pointer-events: none;
    }}
    .ambient {{
        position: fixed; inset:0; z-index:-3;
        background: radial-gradient(600px 220px at 10% 10%, rgba(0,200,190,0.03), transparent 6%),
                    radial-gradient(500px 200px at 90% 80%, rgba(100,70,255,0.02), transparent 6%);
        pointer-events:none;
    }}
    .card {{
        background: linear-gradient(180deg, #ffffff, #fbfdff);
        border-radius:12px;
        border: 1px solid rgba(15,23,36,0.04);
        padding:14px;
        box-shadow: 0 12px 30px rgba(12,18,24,0.06);
    }}
    .muted {{ color:#6b7280; font-size:13px; }}
    .title {{ font-weight:800; font-size:22px; color:#0b1220; }}
    .subtitle {{ color:#6b7280; font-size:13px; margin-top:4px; }}
    .soft-divider {{ height:1px; background:linear-gradient(90deg, rgba(0,0,0,0.03), rgba(0,0,0,0.01)); margin:12px 0; border-radius:2px; }}
    .chip {{ display:inline-block; padding:6px 10px; border-radius:999px; background:#eef2ff; color:#0b1220; font-size:13px; margin-right:6px; border:1px solid rgba(15,23,42,0.03); }}
    .counter {{ font-weight:800; font-size:28px; color:#05204a; }}
    .btn {{ border-radius:8px !important; }}
    @media (max-width:800px) {{
        .title {{ font-size:18px; }}
    }}
    </style>
    <div class="bg-static"></div><div class="ambient"></div>
    """
    st.markdown(css, unsafe_allow_html=True)

# Default background (light, small)
HEART_BG = "https://images.unsplash.com/photo-1542314831-068cd1dbfeeb?q=80&w=1200&auto=format&fit=crop&ixlib=rb-4.0.3&s=5f3f0de6da3d0e6b6bb2f3e3a6f2fb2f"

# ---------------------------
# Utility: load dataset header (to get feature order)
# ---------------------------
def load_feature_list_from_local_or_url(local_path="framingham.csv"):
    if Path(local_path).exists():
        df = pd.read_csv(local_path, nrows=1)
    else:
        # fallback to the common public heart dataset that is similar (but prefer local framingham.csv)
        remote = "https://raw.githubusercontent.com/ishank-j/propublica-tutorials/main/heart.csv"
        try:
            df = pd.read_csv(remote, nrows=1)
        except Exception:
            # create default columns set as final fallback
            cols = ["age","male","education","currentSmoker","cigsPerDay","BPMeds","prevalentStroke","prevalentHyp","diabetes","totChol","sysBP","diaBP","BMI","heartRate","glucose"]
            return cols
    cols = list(df.columns)
    if "TenYearCHD" in cols:
        cols.remove("TenYearCHD")
    else:
        # if target name differs, attempt to remove last column
        cols = cols[:-1]
    return cols

# Determine FEATURE_NAMES exactly as training
FEATURE_NAMES = load_feature_list_from_local_or_url("framingham.csv")

# ---------------------------
# Model paths & helpers
# ---------------------------
MODEL_PATH = "best_heart_chd_model.joblib"
SCALER_PATH = "scaler_chd.joblib"
RANDOM_STATE = 42

def download_and_prepare_full_df(local_path="framingham.csv"):
    # Use local framingham.csv when available (this must be the one you trained on)
    if Path(local_path).exists():
        df = pd.read_csv(local_path)
    else:
        # fallback to remote dataset similar to training (may differ)
        url = "https://raw.githubusercontent.com/ishank-j/propublica-tutorials/main/heart.csv"
        try:
            df = pd.read_csv(url)
        except Exception:
            # synthesize similar dataset if all else fails
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
    # cleaning to mimic your training script
    df = df.replace(["NA","?", " ", ""], np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")
    # fill medians where numeric
    df = df.fillna(df.median(numeric_only=True))
    return df

def train_and_save_model_from_df(df):
    target_col = 'TenYearCHD' if 'TenYearCHD' in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced')
    clf.fit(X_train_s, y_train)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return clf, scaler, list(X.columns), (X_test, y_test)

def load_or_train_model():
    # prefer provided model/scaler files
    if Path(MODEL_PATH).exists() and Path(SCALER_PATH).exists():
        try:
            clf = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            # Build test split from the same df to ensure alignment
            df = download_and_prepare_full_df()
            target_col = 'TenYearCHD' if 'TenYearCHD' in df.columns else df.columns[-1]
            X = df.drop(columns=[target_col])
            y = df[target_col].astype(int)
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.18, random_state=RANDOM_STATE, stratify=y)
            return clf, scaler, list(X.columns), (X_test, y_test)
        except Exception:
            # fallback: retrain to guarantee compatibility
            pass
    # train from available dataset
    df = download_and_prepare_full_df()
    return train_and_save_model_from_df(df)

model, scaler, TRAIN_FEATURE_NAMES, TEST_SPLIT = load_or_train_model()

# ensure FEATURE_NAMES reflect the actual trained order
# prefer TRAIN_FEATURE_NAMES (from trained model) if available
if TRAIN_FEATURE_NAMES and len(TRAIN_FEATURE_NAMES) > 0:
    FEATURE_NAMES = TRAIN_FEATURE_NAMES

# ---------------------------
# Feature alignment (strong)
# ---------------------------
def align_features(df: pd.DataFrame, feature_names: list):
    df = df.copy()
    # add missing
    for c in feature_names:
        if c not in df.columns:
            df[c] = 0.0
    # drop extras & reorder
    df = df[feature_names]
    # ensure numeric dtype
    df = df.astype(float)
    return df

# ---------------------------
# Preprocess / predict / risk label
# ---------------------------
def preprocess_df(df_in: pd.DataFrame):
    return scaler.transform(df_in)

def predict_prob_single(df_in: pd.DataFrame):
    Xs = preprocess_df(df_in)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(Xs)[:,1][0])
    else:
        # fallback: use predict (0/1)
        return float(model.predict(Xs)[0])

def risk_label(prob):
    if prob < 0.15: return "Low"
    if prob < 0.35: return "Moderate"
    if prob < 0.7: return "High"
    return "Very High"

# ---------------------------
# Model accuracy (safe)
# ---------------------------
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
        acc = accuracy_score(y_test, ypred)
        return float(acc)
    except Exception:
        return None

MODEL_ACCURACY = compute_model_accuracy()
ACC_STR = f"{MODEL_ACCURACY:.2%}" if MODEL_ACCURACY is not None else "N/A"

# ---------------------------
# Plot helpers
# ---------------------------
def create_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        title={'text': "10-year CHD (%)"},
        gauge={
            'axis': {'range': [0,100]},
            'bar': {'color': "#ef4444"},
            'steps': [
                {'range':[0,15],'color':'#10b981'},
                {'range':[15,35],'color':'#f59e0b'},
                {'range':[35,70],'color':'#f97316'},
                {'range':[70,100],'color':'#ef4444'}
            ],
            'threshold': {'line': {'color':'#031127','width':4}, 'thickness':0.75, 'value': prob*100}
        }
    ))
    fig.update_layout(height=340, margin=dict(t=10,b=10,l=10,r=10), paper_bgcolor='rgba(0,0,0,0)')
    return fig

def feature_chips(contribs):
    html = "<div style='display:flex; gap:6px; flex-wrap:wrap;'>"
    for f,v in contribs.items():
        html += f"<div class='chip'><strong>{f}</strong>: {v:.2f}</div>"
    html += "</div>"
    return html

# ---------------------------
# UI: Header + Sidebar
# ---------------------------
if "bg_enabled" not in st.session_state:
    st.session_state["bg_enabled"] = True

with st.sidebar:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## Settings")
    st.session_state["bg_enabled"] = st.checkbox("Enable subtle background", value=st.session_state.get("bg_enabled", True))
    st.markdown("---")
    st.markdown("### Presets")
    preset = st.selectbox("Presets", options=["Custom","Healthy demo","High-risk smoker","Elderly hypertensive"])
    if st.button("Clear history"):
        st.session_state['history'] = []
    st.markdown("---")
    st.markdown("### About")
    st.markdown("<div class='muted'>Built by Sachin Ravi — demo ML model. Not for clinical use.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

inject_css(st.session_state["bg_enabled"], HEART_BG)

st.markdown(f"""
<div style="display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:12px;">
  <div style="display:flex; gap:12px; align-items:center;">
    <div style="width:56px; height:56px; border-radius:12px; background:linear-gradient(135deg,#7c3aed,#06b6d4); display:flex; align-items:center; justify-content:center; font-weight:800; color:white;">❤️</div>
    <div>
      <div class="title">Clinical Heart Risk — v3.0</div>
      <div class="subtitle">Built by Sachin Ravi • HealthTech Pro</div>
    </div>
  </div>
  <div style="text-align:right;">
    <div class="muted">Model accuracy: <strong>{ACC_STR}</strong></div>
    <div style="margin-top:6px;"><button class="btn">📥 Download report</button></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Input form (vertical stack, simple keys)
# ---------------------------
col_left, col_right = st.columns([1.4,1], gap="large")

with col_left:
    with st.form("patient_form"):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Patient details")
        age = st.number_input("Age", min_value=18, max_value=120, value=58, key="w_age")
        gender = st.selectbox("Gender", options=["Male","Female"], index=0, key="w_gender")
        education = st.selectbox("Education (1-4)", options=[1,2,3,4], index=1, key="w_edu")
        smoker = st.selectbox("Current smoker?", options=["Smoker","Non-smoker"], index=0, key="w_smoker")
        cigsPerDay = st.number_input("Cigarettes per day", min_value=0, max_value=100, value=20, key="w_cigs")
        BPMeds = st.selectbox("BP medication?", options=["Yes","No"], index=0, key="w_bpmeds")
        prevalentStroke = st.selectbox("Stroke history?", options=["Yes","No"], index=0, key="w_stroke")
        prevalentHyp = st.selectbox("Hypertension?", options=["Yes","No"], index=0, key="w_hyp")
        diabetes = st.selectbox("Diabetes?", options=["Yes","No"], index=0, key="w_diab")
        totChol = st.number_input("Total cholesterol", min_value=100.0, max_value=600.0, value=250.0, key="w_chol")
        sysBP = st.number_input("Systolic BP", min_value=80.0, max_value=240.0, value=160.0, key="w_sys")
        diaBP = st.number_input("Diastolic BP", min_value=40.0, max_value=140.0, value=95.0, key="w_dia")
        BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=29.5, key="w_bmi")
        heartRate = st.number_input("Heart Rate", min_value=30.0, max_value=200.0, value=90.0, key="w_hr")
        glucose = st.number_input("Glucose", min_value=40.0, max_value=400.0, value=140.0, key="w_gl")
        st.markdown("</div>", unsafe_allow_html=True)

        submit = st.form_submit_button("Predict Risk")

# Right column: quick actions & explainability / history area
with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Quick actions")
    st.markdown(f"<div class='muted'>Model accuracy: <strong>{ACC_STR}</strong></div>", unsafe_allow_html=True)
    if st.button("Download last result (PDF)"):
        st.info("Use the predict button to generate a report (after prediction).")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Explainability")
    st.markdown("<div class='muted'>SHAP primary, permutation fallback (aligned to training features)</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# session history
if "history" not in st.session_state:
    st.session_state["history"] = []

# helper: convert UI picks to numeric vector consistent with training
def ui_to_input_dict():
    # Map strings to numbers consistent with training
    data = {}
    # Use FEATURE_NAMES order
    # Build a dict for common expected names (fallback if dataset column names differ)
    # We'll attempt to map many common naming variants
    # Known trained features likely include names from framingham.csv; adapt accordingly.
    # For safety, create a local mapping from our widget names to typical column names:
    mapping = {
        "age": age,
        "male": 1 if gender == "Male" else 0,
        "education": education,
        "currentSmoker": 1 if smoker == "Smoker" else 0,
        "cigsPerDay": cigsPerDay,
        "BPMeds": 1 if BPMeds == "Yes" else 0,
        "prevalentStroke": 1 if prevalentStroke == "Yes" else 0,
        "prevalentHyp": 1 if prevalentHyp == "Yes" else 0,
        "diabetes": 1 if diabetes == "Yes" else 0,
        "totChol": totChol,
        "sysBP": sysBP,
        "diaBP": diaBP,
        "BMI": BMI,
        "heartRate": heartRate,
        "glucose": glucose
    }
    # Build final dict following FEATURE_NAMES order; if name not in mapping, set 0.0
    for fn in FEATURE_NAMES:
        if fn in mapping:
            data[fn] = float(mapping[fn])
        else:
            # try alternative name matches
            low = fn.lower()
            if low in mapping:
                data[fn] = float(mapping[low])
            else:
                data[fn] = 0.0
    return data

# ---------------------------
# Prediction flow (only after submit)
# ---------------------------
if submit:
    # build input dict from widgets (guaranteed fresh values)
    input_dict = ui_to_input_dict()
    # create df and align features (guarantees same order/names)
    X_user = pd.DataFrame([input_dict])
    X_user_aligned = align_features(X_user, FEATURE_NAMES)

    # debug: optionally show the input row (uncomment during dev)
    # st.write("DEBUG - model input (aligned):", X_user_aligned)

    # predict
    try:
        prob = predict_prob_single(X_user_aligned)
        label = risk_label(prob)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # save history
    rec = {"ts": datetime.now().isoformat(timespec='seconds'), "inputs": input_dict, "prob": prob, "label": label}
    st.session_state["history"].insert(0, rec)
    st.session_state["history"] = st.session_state["history"][:40]

    # display results
    st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
    rcol1, rcol2 = st.columns([1.5,1], gap="large")
    with rcol1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"<div class='muted'>Risk probability</div>", unsafe_allow_html=True)
        st.plotly_chart(create_gauge(prob), use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"<div style='text-align:center; font-weight:700; margin-top:8px;'>{prob*100:.1f}% — {label}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with rcol2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div class='muted'>Recommendation</div>", unsafe_allow_html=True)
        if prob > 0.7:
            st.markdown("<div style='font-weight:700;'>Urgent review recommended</div>", unsafe_allow_html=True)
            st.write("Recommend: ECG, lipid panel, fasting glucose, cardiology consult.")
        elif prob > 0.35:
            st.markdown("<div style='font-weight:700;'>High risk — consider follow-up</div>", unsafe_allow_html=True)
            st.write("Lifestyle changes, BP control, further testing.")
        else:
            st.markdown("<div style='font-weight:700;'>Low risk — maintain screening</div>", unsafe_allow_html=True)
            st.write("Lifestyle, routine checks.")
        st.markdown("</div>", unsafe_allow_html=True)

    # explainability block
    st.markdown("<div style='margin-top:12px;' class='card'>", unsafe_allow_html=True)
    st.markdown("### Explainability")
    use_perm = False
    try:
        Xs = preprocess_df(X_user_aligned)
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
            # show top 6
            top = dict(sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6])
            st.markdown(feature_chips(top), unsafe_allow_html=True)
            df_sh = pd.DataFrame({"feature": list(top.keys()), "impact": [top[k] for k in top]})
            st.plotly_chart(px.bar(df_sh, x="impact", y="feature", orientation="h", height=260), use_container_width=True)
        except Exception as e:
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
                # fallback full df
                df_all = download_and_prepare_full_df()
                target_col = 'TenYearCHD' if 'TenYearCHD' in df_all.columns else df_all.columns[-1]
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
            st.info("Explainability unavailable.")

    st.markdown("</div>", unsafe_allow_html=True)

    # PDF export link
    def fig_to_bytes(fig, fmt="png"):
        try:
            return fig.to_image(format=fmt, scale=2)
        except Exception:
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            return buf.read()

    def build_pdf_bytes(inputs, prob, label):
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
            fig = create_gauge(prob)
            img = fig_to_bytes(fig)
            tmp = "tmp_gauge.png"
            with open(tmp, "wb") as f:
                f.write(img)
            pdf.image(tmp, w=160)
        except Exception:
            pdf.cell(0, 6, "Gauge image unavailable", ln=True)
        return pdf.output(dest="S").encode("latin-1")

    pdf_bytes = build_pdf_bytes(input_dict, prob, label)
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="heart_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf">📥 Download PDF Report</a>'
    st.markdown(href, unsafe_allow_html=True)

# ---------------------------
# Session history preview
# ---------------------------
st.markdown("<div style='margin-top:14px;' class='card'>", unsafe_allow_html=True)
st.markdown("### Recent predictions")
if st.session_state["history"]:
    hist_df = pd.DataFrame([{"time":h["ts"], "prob":f'{h["prob"]:.3f}', "label":h["label"]} for h in st.session_state["history"]])
    st.table(hist_df.head(8))
else:
    st.markdown("<div class='muted'>No predictions yet. Use the form to predict.</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div style='margin-top:10px; color:#6b7280; font-size:13px;'>Demo ML model. Not for clinical use. Validate dataset & model before production.</div>", unsafe_allow_html=True)
