# app.py — CHD Predictor v5.1 (Premium; Dark/Light toggle + stethoscope logo)
# Built by Sachin Ravi — v5.1 Premium (Stethoscope Icon)
# Requirements: streamlit, numpy, pandas, scikit-learn, joblib, plotly, matplotlib,
# fpdf, shap (optional), kaleido
# Place model artifacts (best_heart_chd_model.joblib, scaler_chd.joblib, feature_order.json) in repo.

import os, io, json, base64
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from fpdf import FPDF
import plotly.graph_objects as go
import plotly.express as px

# SHAP optional
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# -------------------------
# Config & paths
# -------------------------
st.set_page_config(page_title="CHD Predictor — v5.1", layout="wide", initial_sidebar_state="collapsed")
ROOT = Path.cwd()
MODEL_PATH = ROOT / "best_heart_chd_model.joblib"
SCALER_PATH = ROOT / "scaler_chd.joblib"
FEATURE_ORDER_PATH = ROOT / "feature_order.json"
CSV_PATH = ROOT / "framingham.csv"
FONT_PATH = ROOT / "fonts" / "DejaVuSans.ttf"
RANDOM_STATE = 42

# -------------------------
# Helpers: data & model load (cached)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load model, scaler, and feature order. If missing, attempt lightweight train from csv."""
    # If all artifacts exist, load them
    if MODEL_PATH.exists() and SCALER_PATH.exists() and FEATURE_ORDER_PATH.exists():
        try:
            clf = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            with open(FEATURE_ORDER_PATH, "r") as f:
                features = json.load(f)
            # prepare test split for permutation importance
            df = load_and_clean_df()
            target_col = "TenYearCHD" if "TenYearCHD" in df.columns else df.columns[-1]
            X = df.drop(columns=[target_col])
            y = df[target_col].astype(int)
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.18, random_state=RANDOM_STATE, stratify=y)
            return clf, scaler, features, (X_test, y_test)
        except Exception as e:
            # fallthrough to retrain
            st.warning("Artifact load failed; retraining model from CSV as fallback.")
    # retrain from CSV (safe fallback)
    df = load_and_clean_df()
    if df is None or df.shape[0] < 50:
        raise FileNotFoundError("framingham.csv missing or too small; ensure artifacts exist or provide dataset.")
    target_col = "TenYearCHD" if "TenYearCHD" in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=250, random_state=RANDOM_STATE, class_weight="balanced")
    clf.fit(X_train_s, y_train)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    features = list(X.columns)
    with open(FEATURE_ORDER_PATH, "w") as f:
        json.dump(features, f, indent=2)
    return clf, scaler, features, (X_test, y_test)

@st.cache_data(show_spinner=False)
def load_and_clean_df():
    """Load and clean framingham.csv if present, else None."""
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
    else:
        return None
    # safe numeric conversion and median fill
    df = df.replace(["NA","?"," ", ""], np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(df.median(numeric_only=True))
    return df

# load artifacts
try:
    model, scaler, FEATURE_NAMES, TEST_SPLIT = load_artifacts()
except Exception as e:
    st.error(f"Model artifacts missing and retrain not possible: {e}")
    st.stop()

# ensure feature list valid
if not isinstance(FEATURE_NAMES, list) or len(FEATURE_NAMES) < 5:
    st.error("feature_order.json invalid. Re-run training script to regenerate artifacts.")
    st.stop()

# -------------------------
# Utility: alignment, predict, risk label
# -------------------------
def align_features(df: pd.DataFrame, feature_names: list):
    df = df.copy()
    for c in feature_names:
        if c not in df.columns:
            df[c] = 0.0
    df = df[feature_names]
    return df.astype(float)

def preprocess_and_predict(df_aligned):
    Xs = scaler.transform(df_aligned)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(Xs)[:,1][0])
    return float(model.predict(Xs)[0])

def risk_label(p):
    if p < 0.15: return "Low"
    if p < 0.35: return "Moderate"
    if p < 0.7: return "High"
    return "Very High"

# compute accuracy once (cached)
@st.cache_data(show_spinner=False)
def compute_accuracy():
    try:
        X_test, y_test = TEST_SPLIT
        X_test_al = align_features(X_test, FEATURE_NAMES)
        Xs = scaler.transform(X_test_al)
        try:
            preds = model.predict(Xs)
        except Exception:
            preds = (model.predict_proba(Xs)[:,1] >= 0.5).astype(int)
        return float(accuracy_score(y_test, preds))
    except Exception:
        return None

MODEL_ACCURACY = compute_accuracy()
ACC_STR = f"{MODEL_ACCURACY:.2%}" if MODEL_ACCURACY is not None else "N/A"

# -------------------------
# Theme CSS (dynamic)
# -------------------------
# session theme init
if "theme" not in st.session_state:
    st.session_state.theme = "dark"  # default

def get_css(theme):
    if theme == "light":
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
        :root { --bg: #F7FBFF; --card: rgba(255,255,255,0.9); --text: #111827; --muted: #475569; --accent: #0077FF; --glass: rgba(255,255,255,0.7);}
        html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, "Helvetica Neue", Arial; background: var(--bg); color: var(--text); }
        .topbar { display:flex; justify-content:space-between; align-items:center; padding:12px 8px; margin-bottom:6px; }
        .logo { width:48px; height:48px; border-radius:12px; background: linear-gradient(135deg, #6c5ce7, #00a3ff); display:flex; align-items:center; justify-content:center; color:white; font-weight:700; box-shadow: 0 6px 20px rgba(10,20,30,0.06);}
        .card { background: var(--card); border-radius:12px; padding:14px; box-shadow: 0 10px 30px rgba(2,6,23,0.06); }
        .section-title { font-weight:700; color: var(--text); }
        .muted { color: var(--muted); }
        .pill-low{background:#ecfdf5;color:#059669;padding:6px 10px;border-radius:999px;}
        .pill-moderate{background:#fff7ed;color:#b45309;padding:6px 10px;border-radius:999px;}
        .pill-high{background:#fff1f2;color:#b91c1c;padding:6px 10px;border-radius:999px;}
        </style>
        """
    else:
        # dark
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
        :root { --bg: #071122; --card: rgba(255,255,255,0.03); --text: #e6eef8; --muted: #98a1b3; --accent: #00e0c8; --glass: rgba(255,255,255,0.02); }
        html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, "Helvetica Neue", Arial; background: linear-gradient(180deg,var(--bg), #0b1320); color: var(--text); }
        .topbar { display:flex; justify-content:space-between; align-items:center; padding:12px 8px; margin-bottom:6px; }
        .logo { width:48px; height:48px; border-radius:12px; background: linear-gradient(135deg, #06b6d4, #7c3aed); display:flex; align-items:center; justify-content:center; color:white; font-weight:700; box-shadow: 0 8px 30px rgba(108,92,231,0.14);}
        .card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:12px; padding:14px; border: 1px solid rgba(255,255,255,0.02); box-shadow: 0 10px 30px rgba(2,6,23,0.6); }
        .section-title { font-weight:700; color: var(--text); }
        .muted { color: var(--muted); }
        .pill-low{background:rgba(16,185,129,0.12);color:#10b981;padding:6px 10px;border-radius:999px;}
        .pill-moderate{background:rgba(245,158,11,0.08);color:#f59e0b;padding:6px 10px;border-radius:999px;}
        .pill-high{background:rgba(239,68,68,0.06);color:#ef4444;padding:6px 10px;border-radius:999px;}
        </style>
        """

# inject css
st.markdown(get_css(st.session_state.theme), unsafe_allow_html=True)

# -------------------------
# Topbar with stethoscope logo and theme toggle
# -------------------------
col1, col2 = st.columns([1,1])
with col1:
    # Stethoscope SVG icon inside logo box
    logo_html = """
    <div style="display:flex;align-items:center;gap:10px;">
      <div class="logo">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M20 7V4a1 1 0 0 0-2 0v3a5 5 0 1 1-10 0V4a1 1 0 0 0-2 0v3a7 7 0 1 0 14 0z" fill="white"/>
          <path d="M12 14v6" stroke="white" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <div>
        <div style="font-weight:700;font-size:16px;">Clinical Heart Risk — v5.1</div>
        <div style="font-size:12px;color:var(--muted);">Built by Sachin Ravi • Premium HealthTech</div>
      </div>
    </div>
    """
    st.markdown(logo_html, unsafe_allow_html=True)
with col2:
    # right side: accuracy + theme toggle
    right_html = f'<div style="display:flex;justify-content:flex-end;align-items:center;gap:14px;"><div class="muted">Model accuracy: <strong>{ACC_STR}</strong></div></div>'
    st.markdown(right_html, unsafe_allow_html=True)
    # theme toggle buttons
    tcol = st.columns([1,1])
    with tcol[1]:
        if st.session_state.theme == "dark":
            if st.button("Switch to Light", key="theme_toggle"):
                st.session_state.theme = "light"
                st.experimental_rerun()
        else:
            if st.button("Switch to Dark", key="theme_toggle2"):
                st.session_state.theme = "dark"
                st.experimental_rerun()

# re-inject CSS after theme change to be safe
st.markdown(get_css(st.session_state.theme), unsafe_allow_html=True)

# -------------------------
# Layout Tabs
# -------------------------
tabs = st.tabs(["Predict","Explain","History","Report","About"])
tab_predict, tab_explain, tab_history, tab_report, tab_about = tabs

# -------------------------
# Predict tab content
# -------------------------
with tab_predict:
    # two-column main layout
    left_col, right_col = st.columns([1.1,0.9], gap="large")
    with left_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Patient details</div>', unsafe_allow_html=True)
        # responsive three-column grid implemented using columns
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("👤 Age", min_value=18, max_value=120, value=58, key="age_v51")
            gender = st.selectbox("⚧ Gender", ["Male","Female"], index=0, key="gender_v51")
            education = st.selectbox("🎓 Education (1-4)", [1,2,3,4], index=2, key="edu_v51")
        with c2:
            smoker = st.selectbox("🚬 Current smoker?", ["Smoker","Non-smoker"], index=0, key="smoke_v51")
            cigsPerDay = st.number_input("🚬 Cigarettes / day", min_value=0, max_value=100, value=5, key="cigs_v51")
            BPMeds = st.selectbox("💊 On BP medication?", ["Yes","No"], index=0, key="bpmeds_v51")
        with c3:
            prevalentStroke = st.selectbox("🧠 Stroke history?", ["Yes","No"], index=0, key="stroke_v51")
            prevalentHyp = st.selectbox("⚡ Hypertension?", ["Yes","No"], index=0, key="hyp_v51")
            diabetes = st.selectbox("🩸 Diabetes?", ["Yes","No"], index=0, key="diab_v51")
        # second row vitals
        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
        v1, v2, v3 = st.columns(3)
        with v1:
            totChol = st.number_input("🧪 Total cholesterol (mg/dL)", min_value=100.0, max_value=600.0, value=200.0, key="chol_v51")
            sysBP = st.number_input("🩺 Systolic BP (mmHg)", min_value=80.0, max_value=260.0, value=120.0, key="sys_v51")
        with v2:
            diaBP = st.number_input("🩺 Diastolic BP (mmHg)", min_value=40.0, max_value=160.0, value=80.0, key="dia_v51")
            BMI = st.number_input("⚖️ BMI", min_value=10.0, max_value=60.0, value=25.0, key="bmi_v51")
        with v3:
            heartRate = st.number_input("❤️ Heart Rate (bpm)", min_value=30.0, max_value=200.0, value=72.0, key="hr_v51")
            glucose = st.number_input("🍬 Glucose (mg/dL)", min_value=40.0, max_value=400.0, value=95.0, key="gl_v51")
        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
        predict_btn = st.button("Predict Risk", key="predict_v51")
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Risk Summary</div>', unsafe_allow_html=True)
        gauge_area = st.empty()
        badge_area = st.empty()
        rec_area = st.empty()
        drivers_area = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    # init history
    if "history_v51" not in st.session_state:
        st.session_state.history_v51 = []

    # build input mapping aligned to FEATURE_NAMES
    def build_input():
        m = {
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
        ordered = {fn: m.get(fn, 0.0) for fn in FEATURE_NAMES}
        return ordered

    # predict action
    if predict_btn:
        inputs = build_input()
        df_in = align_features(pd.DataFrame([inputs]), FEATURE_NAMES)
        try:
            prob = preprocess_and_predict(df_in)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            prob = None
        label = risk_label(prob) if prob is not None else "N/A"

        # store history (only essentials)
        st.session_state.history_v51.insert(0, {"ts": datetime.now().isoformat(timespec='seconds'), "prob": prob, "label": label, "inputs": inputs})
        st.session_state.history_v51 = st.session_state.history_v51[:80]

        # render gauge
        if prob is not None:
            # theme-aware colors
            if st.session_state.theme == "light":
                steps = [
                    {'range':[0,15],'color':'#16a34a'},
                    {'range':[15,35],'color':'#f59e0b'},
                    {'range':[35,70],'color':'#fb923c'},
                    {'range':[70,100],'color':'#ef4444'}
                ]
                bar_color = "#0077FF"
            else:
                steps = [
                    {'range':[0,15],'color':'#10b981'},
                    {'range':[15,35],'color':'#f59e0b'},
                    {'range':[35,70],'color':'#fb923c'},
                    {'range':[70,100],'color':'#ef4444'}
                ]
                bar_color = "#00e0c8"

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob*100,
                number={'suffix':'%','font':{'size':44}},
                gauge={
                    'axis':{'range':[0,100],'tickmode':'auto'},
                    'bar':{'color':bar_color, 'thickness':0.26},
                    'steps': steps,
                    'threshold':{'line':{'color':'white','width':4}, 'value':prob*100}
                }
            ))
            fig.update_layout(height=320, margin=dict(t=8,b=8,l=8,r=8), paper_bgcolor="rgba(0,0,0,0)")
            gauge_area.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

            # badge
            if label == "Low":
                badge_area.markdown('<div class="pill-low">Low risk</div>', unsafe_allow_html=True)
            elif label == "Moderate":
                badge_area.markdown('<div class="pill-moderate">Moderate risk</div>', unsafe_allow_html=True)
            else:
                badge_area.markdown('<div class="pill-high">High risk</div>', unsafe_allow_html=True)

            # recs
            if prob > 0.7:
                rec_area.markdown("<strong>Urgent clinical review recommended</strong><ul><li>ECG & cardiology referral</li><li>Fasting glucose & lipid panel</li></ul>", unsafe_allow_html=True)
            elif prob > 0.35:
                rec_area.markdown("<strong>High risk — consider follow-up</strong><ul><li>BP control</li><li>Lifestyle modification</li></ul>", unsafe_allow_html=True)
            else:
                rec_area.markdown("<strong>Low risk — maintain screening</strong><ul><li>Healthy diet & activity</li><li>Routine annual checks</li></ul>", unsafe_allow_html=True)

            # compute top-3 drivers using SHAP or permutation
            drivers_html = build_top3_drivers(df_in)
            drivers_area.markdown(drivers_html, unsafe_allow_html=True)

# -------------------------
# Explain tab
# -------------------------
with tab_explain:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">SHAP explanation attempted first (fast sample). If not available, permutation importance fallback is shown.</div>', unsafe_allow_html=True)

    # prepare test batch sample
    try:
        X_test_full, y_test_full = TEST_SPLIT
    except Exception:
        df_all = load_and_clean_df()
        if df_all is None:
            st.info("No dataset available for explainability sample.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            target_col = 'TenYearCHD' if 'TenYearCHD' in df_all.columns else df_all.columns[-1]
            X_test_full = df_all.drop(columns=[target_col])
            y_test_full = df_all[target_col].astype(int)

    use_perm = False
    plot_placeholder = st.empty()

    # Try SHAP (cached)
    if SHAP_AVAILABLE:
        try:
            shap_fig = compute_shap_plotly(X_test_full)
            plot_placeholder.plotly_chart(shap_fig, use_container_width=True, config={"displayModeBar":True})
        except Exception as e:
            st.warning("SHAP unavailable (or heavy) — switching to permutation importance.")
            use_perm = True
    else:
        use_perm = True

    if use_perm:
        try:
            # sample for speed
            sample = X_test_full.sample(min(200, len(X_test_full)), random_state=RANDOM_STATE)
            y_sample = y_test_full.loc[sample.index]
            res = permutation_importance(model, sample[FEATURE_NAMES], y_sample, n_repeats=6, random_state=RANDOM_STATE, n_jobs=1)
            imp = res.importances_mean
            df_pi = pd.DataFrame({"feature": FEATURE_NAMES, "importance": imp}).sort_values("importance", ascending=True)
            fig_pi = px.bar(df_pi, x="importance", y="feature", orientation="h", height=420)
            fig_pi.update_layout(margin=dict(t=8,b=8,l=120), paper_bgcolor="rgba(0,0,0,0)")
            plot_placeholder.plotly_chart(fig_pi, use_container_width=True, config={"displayModeBar":True})
        except Exception as e:
            st.error(f"Permutation importance failed: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# History tab
# -------------------------
with tab_history:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Recent predictions</div>', unsafe_allow_html=True)
    hist = st.session_state.get("history_v51", [])
    if hist:
        df_hist = pd.DataFrame([{"time":h["ts"], "prob": f'{h["prob"]:.3f}', "label":h["label"]} for h in hist])
        st.dataframe(df_hist, height=300)
        csv = df_hist.to_csv(index=False).encode('utf-8')
        st.markdown(f'<a href="data:file/csv;base64,{base64.b64encode(csv).decode()}" download="predictions_history.csv">Download history CSV</a>', unsafe_allow_html=True)
    else:
        st.markdown("<div class='muted'>No predictions yet. Run the Predict tab to create results.</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Report tab (PDF)
# -------------------------
with tab_report:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Export / Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Generate a professional PDF report for the last prediction.</div>', unsafe_allow_html=True)

    if st.button("Generate PDF of last prediction"):
        hist = st.session_state.get("history_v51", [])
        if not hist:
            st.warning("No prediction yet. Run prediction first.")
        else:
            last = hist[0]
            inputs = last["inputs"]
            prob = last["prob"]
            label = last["label"]
            # try to build importance figure (small)
            try:
                # permutation on sample for PDF
                X_test_full, y_test_full = TEST_SPLIT
                sample = X_test_full.sample(min(120, len(X_test_full)), random_state=RANDOM_STATE)
                y_samp = y_test_full.loc[sample.index]
                res = permutation_importance(model, sample[FEATURE_NAMES], y_samp, n_repeats=6, random_state=RANDOM_STATE, n_jobs=1)
                imp = res.importances_mean
                df_pi = pd.DataFrame({"feature": FEATURE_NAMES, "importance": imp}).sort_values("importance", ascending=False).head(6)
                fig = px.bar(df_pi, x="importance", y="feature", orientation="h", height=220)
                img_bytes = fig.to_image(format="png", scale=2)
            except Exception:
                img_bytes = None

            pdf_bytes = build_pdf_safe(inputs, prob, label, top_importance_fig=img_bytes)
            b64 = base64.b64encode(pdf_bytes).decode()
            st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="chd_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf">📥 Download PDF Report</a>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# About tab
# -------------------------
with tab_about:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">About & Model Card</div>', unsafe_allow_html=True)
    st.markdown(f"""
    **CHD Predictor v5.1** — Premium HealthTech demo built by **Sachin Ravi**.

    **Model**: RandomForestClassifier (exported artifact).  
    **Explainability**: SHAP (TreeExplainer) attempted; permutation importance fallback.  
    **Artifacts**: `{MODEL_PATH.name}`, `{SCALER_PATH.name}`, `{FEATURE_ORDER_PATH.name}`.

    **Important**: This is a demo ML model for educational and portfolio use. **Not** for clinical decision-making.
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# FOOTER
# -------------------------
st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Demo ML model — Not for clinical use. Validate clinically & legally before production.</div>', unsafe_allow_html=True)

# -------------------------
# --- Utility functions used above (placed at bottom for clarity)
# -------------------------
def build_top3_drivers(df_input):
    """
    Returns simple HTML for top 3 drivers using SHAP or permutation fallback.
    df_input: single-row dataframe aligned to FEATURE_NAMES
    """
    try:
        # Try SHAP first (fast single-pred)
        if SHAP_AVAILABLE:
            explainer = shap.TreeExplainer(model)
            Xs = scaler.transform(df_input)
            shap_vals = explainer.shap_values(Xs)
            if isinstance(shap_vals, list):
                vals = shap_vals[1][0]
            else:
                vals = shap_vals[0][0]
            contribs = {FEATURE_NAMES[i]: float(vals[i]) for i in range(len(FEATURE_NAMES))}
            sorted_contribs = sorted(contribs.items(), key=lambda x: -abs(x[1]))[:3]
        else:
            raise Exception("SHAP not available")
    except Exception:
        # fallback permutation importance on small sample (cheap)
        try:
            X_test_full, y_test_full = TEST_SPLIT
            sample = X_test_full.sample(min(200, len(X_test_full)), random_state=RANDOM_STATE)
            res = permutation_importance(model, sample[FEATURE_NAMES], y_test_full.loc[sample.index], n_repeats=6, random_state=RANDOM_STATE, n_jobs=1)
            imp = res.importances_mean
            df_pi = pd.DataFrame({"feature": FEATURE_NAMES, "importance": imp}).sort_values("importance", ascending=False).head(3)
            sorted_contribs = [(r['feature'], float(r['importance'])) for _, r in df_pi.iterrows()]
        except Exception:
            sorted_contribs = []

    # build HTML
    if not sorted_contribs:
        return "<div class='muted'>Top drivers unavailable.</div>"
    html = "<div><strong>Key Risk Drivers</strong><ul>"
    for feat, val in sorted_contribs:
        direction = "↑" if val > 0 else "↓"
        html += f"<li><strong>{feat}</strong> {direction} ({val:.3f})</li>"
    html += "</ul></div>"
    return html

def compute_shap_plotly(X_test):
    """Compute a shap importance bar plotly figure (cached)."""
    # sample
    sample = X_test.sample(min(120, len(X_test)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(sample)
    if isinstance(shap_vals, list):
        shap_mean = np.mean(np.abs(shap_vals[1]), axis=0)
    else:
        shap_mean = np.mean(np.abs(shap_vals), axis=0)
    df_shap = pd.DataFrame({"feature": FEATURE_NAMES, "importance": shap_mean}).sort_values("importance", ascending=True)
    fig = px.bar(df_shap, x="importance", y="feature", orientation="h", height=420)
    fig.update_layout(margin=dict(t=8,b=8,l=120), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def build_pdf_safe(inputs: dict, prob: float, label: str, top_importance_fig=None):
    """
    Build a unicode-safe PDF, using DejaVuSans.ttf if available.
    top_importance_fig: optional PNG bytes to embed
    """
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=12)
    pdf.add_page()
    # register unicode font
    try:
        if FONT_PATH.exists():
            pdf.add_font("DejaVu", "", str(FONT_PATH), uni=True)
            pdf.set_font("DejaVu", size=14)
        else:
            pdf.set_font("Arial", size=14)
    except Exception:
        pdf.set_font("Arial", size=14)

    pdf.cell(0, 8, "CHD Predictor Report — v5.1", ln=True, align="C")
    pdf.ln(4)
    pdf.set_font(size=10)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(6)
    pdf.set_font("", style="B")
    pdf.cell(0, 6, "Inputs", ln=True)
    pdf.set_font("", style="")
    for k, v in inputs.items():
        safe_v = str(v)
        try:
            pdf.cell(0,6, f"{k}: {safe_v}", ln=True)
        except Exception:
            pdf.cell(0,6, f"{k}: {safe_v.encode('utf-8', errors='replace').decode('utf-8')}", ln=True)
    pdf.ln(6)
    pdf.set_font("", style="B")
    pdf.cell(0,6, "Prediction", ln=True)
    pdf.set_font("", style="")
    pdf.cell(0,6, f"Probability: {prob:.3f}", ln=True)
    pdf.cell(0,6, f"Risk level: {label}", ln=True)
    pdf.ln(6)
    if top_importance_fig is not None:
        try:
            tmp = "tmp_imp.png"
            with open(tmp, "wb") as f:
                f.write(top_importance_fig)
            pdf.image(tmp, w=170)
        except Exception:
            pdf.cell(0,6,"Importance image unavailable", ln=True)
    pdf.ln(8)
    pdf.set_font("", size=9)
    pdf.cell(0,6, "Report generated by CHD Predictor — Built by Sachin Ravi", ln=True)
    pdf.cell(0,6, "Demo ML model — Not for clinical use.", ln=True)

    try:
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
    except UnicodeEncodeError:
        pdf_bytes = pdf.output(dest="S").encode("latin-1", errors="replace")
    return pdf_bytes

# End of file
