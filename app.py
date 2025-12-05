# app.py — PREMIUM CLIENT-READY HEART DISEASE DASHBOARD
# Replace your existing app.py with this file (it expects your model files present):
#   best_heart_chd_model.joblib
#   scaler_chd.joblib
# Requirements: streamlit, numpy, pandas, scikit-learn, joblib, plotly, matplotlib, fpdf, shap, kaleido

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io
from fpdf import FPDF
import matplotlib.pyplot as plt
import base64
import sys
import time

# Safe imports related to SHAP — optional fallback if not possible
try:
    import shap
except Exception:
    shap = None

# ---------------------------
# PAGE CONFIG AND STYLING
# ---------------------------
st.set_page_config(page_title="Heart Disease Predictor — Premium", page_icon="❤️", layout="wide")

# Colors & theme
ACCENT = "#ff4b4b"       # primary accent (red)
ACCENT2 = "#ff6b6b"
SAFE = "#16a34a"
CAUTION = "#f59e0b"
DARK_BG = "#0b1220"
CARD_BG = "rgba(255,255,255,0.02)"
TEXT = "#cfe7ff"
MUTED = "#9fb0c9"

# Minimal custom CSS for premium look
st.markdown(
    f"""
    <style>
      :root {{
        --bg: {DARK_BG};
        --card: rgba(255,255,255,0.02);
        --accent: {ACCENT};
        --text: {TEXT};
        --muted: {MUTED};
      }}
      body {{
        background: linear-gradient(180deg,#071223 0%, #0b1220 100%) !important;
        color: var(--text);
      }}
      .stApp > header {{ display: none; }}
      .card {{
        background: var(--card);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.6);
        border: 1px solid rgba(255,255,255,0.03);
        color: var(--text);
      }}
      .small-muted {{ color: var(--muted); font-size:13px; }}
      .kv {{ font-weight:700; color:#dff2ff; }}
      .btn-primary {{
        background: linear-gradient(90deg, {ACCENT2}, {ACCENT}); 
        padding:10px 18px; border-radius:10px; color:white; font-weight:700;
      }}
      .tooltip {{
        color: var(--muted); font-size:13px;
      }}
      .footer {{ color:#7b8ba3; font-size:12px; text-align:center; padding-top:18px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# LOAD MODEL & SCALER (Use your filenames)
# ---------------------------
MODEL_FILENAME = "best_heart_chd_model.joblib"
SCALER_FILENAME = "scaler_chd.joblib"

try:
    model = joblib.load(MODEL_FILENAME)
    scaler = joblib.load(SCALER_FILENAME)
except Exception as e:
    st.error("Model or scaler file not found or failed to load. Make sure the repository contains:\n"
             f"`{MODEL_FILENAME}` and `{SCALER_FILENAME}`. Error: " + str(e))
    st.stop()

# ---------------------------
# UTILS
# ---------------------------
def to_bin(x):
    return 1 if str(x).lower() in ("yes", "male", "1", "true") else 0

def build_input_array(d):
    """
    Build numpy array in the exact feature order used during training.
    Adjust if training order differs.
    """
    arr = np.array([[
        to_bin(d.get("male")),
        d.get("age"),
        d.get("education"),
        to_bin(d.get("currentSmoker")),
        d.get("cigsPerDay"),
        to_bin(d.get("BPMeds")),
        to_bin(d.get("prevalentStroke", 0)),
        to_bin(d.get("prevalentHyp", 0)),
        to_bin(d.get("diabetes")),
        d.get("totChol"),
        d.get("sysBP"),
        d.get("diaBP"),
        d.get("BMI"),
        d.get("heartRate", 72),
        d.get("glucose")
    ]], dtype=float)
    return arr

def make_gauge_plot(prob):
    """Return a Plotly gauge figure for the given probability (0..1)."""
    val = float(prob) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        title={'text': "Risk %"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth':1, 'tickcolor':'#999'},
            'bar': {'color': ACCENT},
            'bgcolor': "rgba(0,0,0,0)",
            'steps': [
                {'range':[0,30], 'color': SAFE},
                {'range':[30,60], 'color': CAUTION},
                {'range':[60,100], 'color': ACCENT}
            ],
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=420, margin=dict(t=30, b=10, l=10, r=10))
    return fig

def make_prob_bar(prob):
    fig = go.Figure(data=[go.Bar(x=["CHD Risk"], y=[prob], marker_color=ACCENT)])
    fig.update_layout(yaxis=dict(range=[0,1], tickformat=".0%"), paper_bgcolor="rgba(0,0,0,0)", height=260, margin=dict(t=10,b=10))
    return fig

def gauge_png_bytes(fig):
    """
    Try to export Plotly figure to PNG bytes using to_image (kaleido).
    If it fails, return None.
    """
    try:
        img = fig.to_image(format="png", width=800, height=480, scale=2)
        return img
    except Exception:
        return None

def pdf_from_report(input_df, pred_label, prob, note="", gauge_png=None):
    """
    Build a simple PDF bytes report containing summary and (optionally) gauge image.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 8, "Heart Disease Risk Report", ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 6, f"Prediction: {pred_label}", ln=True)
    pdf.cell(0, 6, f"Probability: {prob:.2f}", ln=True)
    pdf.ln(4)
    if note:
        pdf.multi_cell(0, 5, f"Note: {note}")
        pdf.ln(4)
    # Insert gauge if available
    if gauge_png:
        # write bytes to a temp file then include
        try:
            tmp_path = "/tmp/gauge.png"
            with open(tmp_path, "wb") as f:
                f.write(gauge_png)
            pdf.image(tmp_path, x=25, w=160)
            pdf.ln(6)
        except Exception:
            pass
    pdf.set_font("Arial", size=10)
    pdf.cell(0,6, "Input values:", ln=True)
    for k,v in input_df.iloc[0].items():
        pdf.multi_cell(0,5, f" - {k}: {v}")
    return pdf.output(dest="S").encode("latin-1")

# SHAP caching helper (if shap is available)
@st.cache_data(show_spinner=False)
def compute_shap_cached(model_obj, background_arr, X_arr):
    # Returns (explainer, shap_values) or raises
    if shap is None:
        raise RuntimeError("SHAP not installed in the environment.")
    try:
        explainer = shap.Explainer(model_obj, background_arr)
        sv = explainer(X_arr)
        return explainer, sv
    except Exception:
        # fallback to kernel
        explainer = shap.KernelExplainer(model_obj.predict_proba, background_arr)
        sv = explainer.shap_values(X_arr, nsamples=100)
        return explainer, sv

# ---------------------------
# SIDEBAR
# ---------------------------
with st.sidebar:
    st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color:{ACCENT}; margin:0;'>❤️ CHD Predictor</h2>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Premium UI • Exportable reports • Explainable AI</div>", unsafe_allow_html=True)
    st.markdown("</div>")
    st.divider = st.markdown("---", unsafe_allow_html=True)
    st.header("Quick controls")
    dark_mode = st.checkbox("🌙 Dark mode", value=True)
    show_header = st.checkbox("Show header", value=True)
    st.markdown("### Presets")
    if st.button("Load: Healthy (Demo)"):
        st.session_state["_preset"] = "healthy"
    if st.button("Load: High-risk smoker (Demo)"):
        st.session_state["_preset"] = "high_smoker"
    st.markdown("---")
    st.markdown("### About")
    st.markdown("Model trained on Framingham dataset. This app is an academic demo, not clinical advice.")
    st.markdown("---")
    st.markdown("<div class='small-muted'>© 2025 • Built by Sachin</div>", unsafe_allow_html=True)

# ---------------------------
# HERO (top area)
# ---------------------------
if st.sidebar.checkbox("Show hero", value=True):
    col1, col2 = st.columns([1,4])
    with col1:
        st.markdown(f"<div style='padding:6px; border-radius:8px; background: linear-gradient(90deg,{ACCENT2},{ACCENT}); display:inline-block'><strong style='color:white; padding:6px'>CHD</strong></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<h1 style='margin-bottom:6px;'>Professional Heart Disease Risk Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<div class='small-muted'>Interactive demo with premium UI, PDF export, and explainability.</div>", unsafe_allow_html=True)
    st.markdown("---")

# ---------------------------
# MAIN UI: Tabs
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Explain", "Report", "About"])

# ---------- TAB: Predict ----------
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Patient details (grouped inputs)", unsafe_allow_html=True)

    # Preset handling
    preset = st.session_state.get("_preset", None)
    if preset == "healthy":
        preset_vals = dict(male="Female", age=45, education=2, currentSmoker="No", cigsPerDay=0, BPMeds="No",
                           prevalentStroke="No", prevalentHyp="No", diabetes="No", totChol=180, sysBP=120, diaBP=78, BMI=23, heartRate=72, glucose=85)
        # clear preset flag
        st.session_state["_preset"] = None
    elif preset == "high_smoker":
        preset_vals = dict(male="Male", age=62, education=1, currentSmoker="Yes", cigsPerDay=20, BPMeds="Yes",
                           prevalentStroke="No", prevalentHyp="Yes", diabetes="Yes", totChol=260, sysBP=150, diaBP=90, BMI=30, heartRate=78, glucose=140)
        st.session_state["_preset"] = None
    else:
        preset_vals = None

    # Form inputs
    with st.form("predict_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=120, value=(preset_vals['age'] if preset_vals else 50))
            gender = st.selectbox("Gender", ["Male", "Female"], index=(0 if (preset_vals and preset_vals['male']=='Male') else 1))
            education = st.number_input("Education (1–4)", 1, 4, value=(preset_vals['education'] if preset_vals else 1))
            currentSmoker = st.selectbox("Current smoker?", ["Yes", "No"], index=(0 if (preset_vals and preset_vals['currentSmoker']=='Yes') else 1))
            cigsPerDay = st.number_input("Cigarettes / day", 0, 100, value=(preset_vals['cigsPerDay'] if preset_vals else 0))
        with c2:
            BPMeds = st.selectbox("On BP medication?", ["Yes", "No"], index=(0 if (preset_vals and preset_vals['BPMeds']=='Yes') else 1))
            prevalentStroke = st.selectbox("Stroke history?", ["Yes", "No"], index=(0 if (preset_vals and preset_vals['prevalentStroke']=='Yes') else 1))
            prevalentHyp = st.selectbox("Hypertension?", ["Yes", "No"], index=(0 if (preset_vals and preset_vals['prevalentHyp']=='Yes') else 1))
            diabetes = st.selectbox("Diabetes?", ["Yes", "No"], index=(0 if (preset_vals and preset_vals['diabetes']=='Yes') else 1))
            BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=(preset_vals['BMI'] if preset_vals else 25.0))
        with c3:
            totChol = st.number_input("Total cholesterol", 100, 600, value=(preset_vals['totChol'] if preset_vals else 200))
            sysBP = st.number_input("Systolic BP", 80, 250, value=(preset_vals['sysBP'] if preset_vals else 120))
            diaBP = st.number_input("Diastolic BP", 40, 200, value=(preset_vals['diaBP'] if preset_vals else 80))
            heartRate = st.number_input("Heart rate", 30, 200, value=(preset_vals['heartRate'] if preset_vals else 72))
            glucose = st.number_input("Glucose", 40, 400, value=(preset_vals['glucose'] if preset_vals else 90))

        submitted = st.form_submit_button("🔍 Predict", use_container_width=False)

    st.markdown("</div>", unsafe_allow_html=True)
    # Submit handling
    if submitted:
        input_vals = {
            "male": gender, "age": age, "education": education, "currentSmoker": currentSmoker,
            "cigsPerDay": cigsPerDay, "BPMeds": BPMeds, "prevalentStroke": prevalentStroke,
            "prevalentHyp": prevalentHyp, "diabetes": diabetes, "totChol": totChol,
            "sysBP": sysBP, "diaBP": diaBP, "BMI": BMI, "heartRate": heartRate, "glucose": glucose
        }

        # Build array -> scale -> predict
        X_arr = build_input_array(input_vals)
        try:
            X_scaled = scaler.transform(X_arr)
        except Exception as e:
            st.error("Scaling failed: " + str(e))
            X_scaled = X_arr

        try:
            pred = model.predict(X_scaled)[0]
            prob = float(model.predict_proba(X_scaled)[0][1])
        except Exception as e:
            st.error("Model prediction failed: " + str(e))
            st.stop()

        # Save to session
        st.session_state["last_input"] = input_vals
        st.session_state["last_pred"] = int(pred)
        st.session_state["last_prob"] = float(prob)

        # Top summary row: risk badge, recommendation, export
        colA, colB, colC = st.columns([1.5, 2, 1])
        with colA:
            level = "LOW" if prob < 0.30 else ("MEDIUM" if prob < 0.60 else "HIGH")
            color = SAFE if prob < 0.30 else (CAUTION if prob < 0.60 else ACCENT)
            st.markdown(f"<div class='card'><h3 style='color:{color}; margin:0;'>Risk: <span style='font-size:24px'>{level}</span></h3><div class='small-muted'>Probability: {prob:.2f}</div></div>", unsafe_allow_html=True)
        with colB:
            st.markdown("<div class='card'><div class='kv'>Recommendation</div>", unsafe_allow_html=True)
            if pred == 1:
                st.markdown("<b>Consult a doctor urgently. Recommended tests: ECG, lipid profile, fasting glucose.</b>", unsafe_allow_html=True)
            else:
                st.markdown("Maintain healthy lifestyle; routine checkups annually recommended.", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with colC:
            st.markdown("<div class='card'><div class='kv'>Export</div>", unsafe_allow_html=True)
            # Build gauge fig and try to embed PNG for PDF
            gauge_fig = make_gauge_plot(prob)
            gauge_png = gauge_png_bytes(gauge_fig)  # might return None if kaleido missing
            pdf_bytes = pdf_from_report(pd.DataFrame([input_vals]), ("HIGH" if pred==1 else "LOW"), prob, note="Academic demo; not medical advice.", gauge_png=gauge_png)
            st.download_button("📥 Download PDF Report", data=pdf_bytes, file_name="chd_report.pdf", mime="application/pdf")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        # Visuals row
        v1, v2 = st.columns([1.6, 1])
        with v1:
            st.plotly_chart(gauge_fig, use_container_width=True)
        with v2:
            st.plotly_chart(make_prob_bar(prob), use_container_width=True)

        st.markdown("### Model input preview")
        preview_df = pd.DataFrame([input_vals]).T.rename(columns={0:"value"})
        st.dataframe(preview_df, height=260)

        # show small positive micro-interaction
        if prob < 0.3:
            st.balloons()
        elif prob > 0.85:
            st.snow()

# ---------- TAB: Explain ----------
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Model explainability (SHAP)")
    st.markdown("Compute SHAP values to understand which features pushed the model toward this prediction. (May be slow.)", unsafe_allow_html=True)

    if "last_input" not in st.session_state:
        st.info("Run a prediction in the Predict tab first.")
    else:
        if shap is None:
            st.warning("SHAP library not installed. Add `shap` to requirements.txt if you want SHAP plots.")
        else:
            if st.button("Compute SHAP (may take time)"):
                with st.spinner("Computing SHAP values..."):
                    try:
                        # build small background (repeat current input if training data unavailable)
                        bg = np.repeat(np.array(list(st.session_state["last_input"].values()), dtype=float).reshape(1,-1), 50, axis=0)
                        explainer, sv = compute_shap_cached(model, bg, np.array([list(st.session_state["last_input"].values())], dtype=float))
                        # try waterfall
                        try:
                            st.set_option('deprecation.showPyplotGlobalUse', False)
                            fig = shap.plots.waterfall(sv[0], show=False)
                            st.pyplot(bbox_inches='tight')
                        except Exception:
                            # fallback to bar chart of mean absolute shap values
                            vals = np.abs(sv.values).mean(0) if hasattr(sv, "values") else np.mean(np.abs(sv[0]), axis=0)
                            feat_names = list(st.session_state["last_input"].keys())
                            fig2, ax = plt.subplots(figsize=(6,4))
                            ax.barh(feat_names, vals)
                            ax.set_xlabel("Mean |SHAP value|")
                            st.pyplot(fig2)
                    except Exception as e:
                        st.error("SHAP failed: " + str(e))
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- TAB: Report ----------
with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Deliverables & Checklist", unsafe_allow_html=True)
    st.markdown("- Download PDF from Predict tab for your submission.")
    st.markdown("- Save screenshots: EDA, model metrics, SHAP, app UI.")
    st.markdown("- Add live app link to your paper and README.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- TAB: About ----------
with tab4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### About this project")
    st.markdown("""
    **Dataset:** Framingham Heart Study (public)  
    **What this app does:** Predict 10-year CHD risk, export report, and offer explainability via SHAP.  
    **Disclaimer:** This is an academic demo only — not a medical diagnostic tool.
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>Built by Sachin • For academic use • Add this link to your submission</div>", unsafe_allow_html=True)
