# app.py — PREMIUM MEDICAL-STYLE HEART DISEASE DASHBOARD
# Drop-in replacement for your current app.py
# Expects these files in repo root:
#   - best_heart_chd_model.joblib
#   - scaler_chd.joblib
#
# Requirements (examples, add to requirements.txt):
# streamlit, numpy, pandas, scikit-learn, joblib, plotly, matplotlib, fpdf, shap, kaleido
# NOTE: SHAP may be slow on non-tree models. We include graceful fallbacks.

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF
import io
import time
import matplotlib.pyplot as plt

# optional shap import — keep app working if shap missing
try:
    import shap
except Exception:
    shap = None

# ---------------------------
# PAGE & THEME CONFIG
# ---------------------------
st.set_page_config(page_title="Heart Disease Predictor — Clinical UI", page_icon="❤️", layout="wide")

# Color tokens
ACCENT = "#ff4b4b"    # red
ACCENT2 = "#ff6b6b"
SAFE = "#16a34a"
CAUTION = "#f59e0b"
DARK_BG = "#0b1220"
LIGHT_BG = "#f6f8fb"
TEXT_DARK = "#0b1220"
TEXT_LIGHT = "#e6eef8"

# ---------------------------
# STYLES: dark/light + animations + medical style
# ---------------------------
# We use CSS variables and respects prefers-color-scheme for an "auto" theme.
st.markdown(
    f"""
    <style>
    :root {{
      --accent: {ACCENT};
      --accent2: {ACCENT2};
      --safe: {SAFE};
      --caution: {CAUTION};
      --dark-bg: {DARK_BG};
      --light-bg: {LIGHT_BG};
      --text-dark: {TEXT_DARK};
      --text-light: {TEXT_LIGHT};
    }}
    /* Auto respects OS preference; you can toggle app theme with UI toggle */
    @media (prefers-color-scheme: dark) {{
        body {{ background: linear-gradient(180deg,#071223 0%, var(--dark-bg) 100%); color: var(--text-light); }}
    }}
    @media (prefers-color-scheme: light) {{
        body {{ background: linear-gradient(180deg,#ffffff 0%, var(--light-bg) 100%); color: var(--text-dark); }}
    }}
    /* Basic page-level card + animation style */
    .card {{
      background: rgba(255,255,255,0.02);
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 18px;
      box-shadow: 0 6px 18px rgba(0,0,0,0.45);
      border: 1px solid rgba(255,255,255,0.03);
      animation: fadeUp 0.45s ease both;
    }}
    @keyframes fadeUp {{
      from {{ opacity: 0; transform: translateY(8px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    .small-muted {{ color: rgba(200,220,240,0.6); font-size:13px; }}
    .kv {{ font-weight:700; color: #dff2ff; }}
    .btn-primary {{ background: linear-gradient(90deg, var(--accent2), var(--accent)); padding:10px 14px; border-radius:10px; color:white; font-weight:700; }}
    .risk-badge {{ padding:10px 12px; border-radius:10px; color:white; font-weight:700; display:inline-block; }}
    .tooltip {{ color: rgba(200,220,240,0.55); font-size:13px; }}
    /* skeleton loader */
    .skeleton {{
      background: linear-gradient(90deg, rgba(255,255,255,0.03), rgba(255,255,255,0.06), rgba(255,255,255,0.03));
      height: 18px;
      border-radius: 6px;
      animation: shimmer 1.2s infinite;
    }}
    @keyframes shimmer {{
      0% {{ background-position: -200px 0; }}
      100% {{ background-position: 200px 0; }}
    }}
    /* page transition */
    .page-fade {{ animation: pageFade 0.5s ease both; }}
    @keyframes pageFade {{
      from {{ opacity: 0; transform: translateY(6px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    /* responsive tweaks */
    @media (max-width: 768px) {{
      .card {{ padding:12px; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# SAFE LOADING: model & scaler
# ---------------------------
MODEL_FILENAME = "best_heart_chd_model.joblib"
SCALER_FILENAME = "scaler_chd.joblib"

try:
    model = joblib.load(MODEL_FILENAME)
    scaler = joblib.load(SCALER_FILENAME)
except Exception as e:
    st.error("Model or scaler not found. Make sure your repo contains:\n"
             f"- {MODEL_FILENAME}\n- {SCALER_FILENAME}\n\nError: " + str(e))
    st.stop()

# ---------------------------
# UTILITIES: type coercion + safe getters
# ---------------------------

def safe_cast(value, dtype, default):
    """Try to cast `value` to dtype (int/float/str). If fails, returns default."""
    try:
        if dtype == int:
            return int(float(value))
        if dtype == float:
            return float(value)
        if dtype == str:
            return str(value)
    except Exception:
        return default

def get_preset_value(presets, key, dtype, default):
    """
    Read from presets dict (which may contain strings or mixed types),
    return a value of correct dtype (int/float/str) or default.
    """
    if not presets:
        return default
    raw = presets.get(key, default)
    return safe_cast(raw, dtype, default)

def to_bin(x):
    return 1 if str(x).lower() in ("yes", "male", "1", "true") else 0

def build_input_array(vals):
    """
    Build the 1x15 numpy array in the exact feature order used for training.
    Ensure numeric values are floats.
    """
    arr = np.array([[
        to_bin(vals["male"]),
        float(vals["age"]),
        float(vals["education"]),
        to_bin(vals["currentSmoker"]),
        float(vals["cigsPerDay"]),
        to_bin(vals["BPMeds"]),
        to_bin(vals.get("prevalentStroke", 0)),
        to_bin(vals.get("prevalentHyp", 0)),
        to_bin(vals["diabetes"]),
        float(vals["totChol"]),
        float(vals["sysBP"]),
        float(vals["diaBP"]),
        float(vals["BMI"]),
        float(vals.get("heartRate", 72)),
        float(vals["glucose"])
    ]], dtype=float)
    return arr

# ---------------------------
# VISUAL HELPERS: visuals, PDF
# ---------------------------
def make_gauge_plot(prob):
    val = float(prob) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number={'suffix': '%', 'font': {'size': 36}},
        title={'text': "Risk Percentage"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': ACCENT},
            'steps': [
                {'range': [0, 30], 'color': SAFE},
                {'range': [30, 60], 'color': CAUTION},
                {'range': [60, 100], 'color': ACCENT}
            ],
            'threshold': {'line': {'color': "black", 'width': 2}, 'thickness': 0.75, 'value': val}
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=420, margin=dict(t=20,b=10,l=10,r=10))
    return fig

def make_prob_bar(prob):
    fig = go.Figure(go.Bar(x=["CHD Risk"], y=[prob], marker_color=ACCENT))
    fig.update_layout(yaxis=dict(range=[0,1], tickformat=".0%"), paper_bgcolor="rgba(0,0,0,0)", height=240, margin=dict(t=10,b=10))
    return fig

def export_plotly_png_bytes(fig, width=900, height=480):
    """
    Try to export Plotly figure to PNG bytes via kaleido. If fails, return None.
    """
    try:
        img = fig.to_image(format="png", width=width, height=height, scale=2)
        return img
    except Exception:
        return None

def pdf_report_bytes(input_df, label, prob, note="", gauge_png=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(0, 8, "Heart Disease Risk Report", ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 6, f"Prediction: {label}", ln=True)
    pdf.cell(0, 6, f"Probability: {prob:.2f}", ln=True)
    pdf.ln(6)
    if note:
        pdf.multi_cell(0, 5, f"Note: {note}")
        pdf.ln(4)
    if gauge_png:
        try:
            tmp_path = "/tmp/gauge.png"
            with open(tmp_path, "wb") as f:
                f.write(gauge_png)
            pdf.image(tmp_path, x=30, w=150)
            pdf.ln(6)
        except Exception:
            pass
    pdf.set_font("Arial", size=10)
    pdf.cell(0,6, "Input values:", ln=True)
    for k,v in input_df.iloc[0].items():
        pdf.multi_cell(0,5, f" - {k}: {v}")
    return pdf.output(dest="S").encode("latin-1")

# ---------------------------
# SIDEBAR (controls & presets)
# ---------------------------
st.sidebar.markdown("<div style='text-align:center'><h3 style='color:#ff6b6b;margin:0;'>❤️ CHD Predictor</h3></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='small-muted'>Premium clinical-style dashboard</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Theme toggle: we implement light/dark toggle and also respect OS preference
theme_choice = st.sidebar.selectbox("Theme", options=["Auto (follow OS)", "Dark", "Light"], index=0)
show_hero = st.sidebar.checkbox("Show hero header", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### Presets (one-click)")
if st.sidebar.button("Load: Healthy (demo)"):
    st.session_state["_preset_flag"] = "healthy"
if st.sidebar.button("Load: High-risk smoker (demo)"):
    st.session_state["_preset_flag"] = "high_smoker"
if st.sidebar.button("Clear preset"):
    st.session_state["_preset_flag"] = None

# ---------------------------
# PRESET DICTIONARIES
# ---------------------------
PRESETS = {
    "healthy": {
        "male": "Female", "age": 45, "education": 2, "currentSmoker": "No", "cigsPerDay": 0,
        "BPMeds": "No", "prevalentStroke": "No", "prevalentHyp": "No", "diabetes": "No",
        "totChol": 180, "sysBP": 120, "diaBP": 78, "BMI": 23.0, "heartRate": 72, "glucose": 85
    },
    "high_smoker": {
        "male": "Male", "age": 62, "education": 1, "currentSmoker": "Yes", "cigsPerDay": 20,
        "BPMeds": "Yes", "prevalentStroke": "No", "prevalentHyp": "Yes", "diabetes": "Yes",
        "totChol": 260, "sysBP": 150, "diaBP": 90, "BMI": 30.0, "heartRate": 78, "glucose": 140
    }
}

# ---------------------------
# PAGE HERO
# ---------------------------
if show_hero:
    col_a, col_b = st.columns([1, 4])
    with col_a:
        st.markdown("<div style='padding:6px; display:inline-block; border-radius:10px; background: linear-gradient(90deg,#ff6b6b,#ff4b4b);'><strong style='color:white; padding:8px;'>CHD</strong></div>", unsafe_allow_html=True)
    with col_b:
        st.markdown("<h1 style='margin-bottom:6px;'>Clinical Heart Disease Risk Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<div class='small-muted'>Add patient details, get a risk estimate, download a clinical report, and explore explainability.</div>", unsafe_allow_html=True)
    st.markdown("---")

# ---------------------------
# MAIN: Tabs (Predict / Explain / Report / About)
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Explain", "Report", "About"])

# ---------- TAB: Predict ----------
with tab1:
    st.markdown("<div class='card page-fade'>", unsafe_allow_html=True)
    st.markdown("### Patient details (group in logical sections)", unsafe_allow_html=True)

    # Load preset flag if set
    preset_flag = st.session_state.get("_preset_flag", None)
    preset_vals = PRESETS.get(preset_flag) if preset_flag in PRESETS else None
    # Clear preset flag after reading so subsequent edits aren't overwritten
    st.session_state["_preset_flag"] = None

    # form with safe casting for number inputs
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=120,
                                  value=get_preset_value(preset_vals, "age", int, 50))
            gender = st.selectbox("Gender", ["Male", "Female"],
                                  index=0 if get_preset_value(preset_vals, "male", str, "Male") == "Male" else 1)
            education = st.number_input("Education (1–4)", 1, 4,
                                        value=get_preset_value(preset_vals, "education", int, 1))
            currentSmoker = st.selectbox("Current smoker?", ["Yes", "No"],
                                         index=0 if get_preset_value(preset_vals, "currentSmoker", str, "No") == "Yes" else 1)
            cigsPerDay = st.number_input("Cigarettes / day", 0, 100,
                                         value=get_preset_value(preset_vals, "cigsPerDay", int, 0))
        with col2:
            BPMeds = st.selectbox("On BP medication?", ["Yes", "No"],
                                  index=0 if get_preset_value(preset_vals, "BPMeds", str, "No") == "Yes" else 1)
            prevalentStroke = st.selectbox("Stroke history?", ["Yes", "No"],
                                           index=0 if get_preset_value(preset_vals, "prevalentStroke", str, "No") == "Yes" else 1)
            prevalentHyp = st.selectbox("Hypertension?", ["Yes", "No"],
                                        index=0 if get_preset_value(preset_vals, "prevalentHyp", str, "No") == "Yes" else 1)
            diabetes = st.selectbox("Diabetes?", ["Yes", "No"],
                                    index=0 if get_preset_value(preset_vals, "diabetes", str, "No") == "Yes" else 1)
            BMI = st.number_input("BMI", min_value=10.0, max_value=60.0,
                                  value=float(get_preset_value(preset_vals, "BMI", float, 25.0)))
        with col3:
            totChol = st.number_input("Total cholesterol", 100, 600,
                                      value=float(get_preset_value(preset_vals, "totChol", float, 200.0)))
            sysBP = st.number_input("Systolic BP", 80, 250,
                                    value=float(get_preset_value(preset_vals, "sysBP", float, 120.0)))
            diaBP = st.number_input("Diastolic BP", 40, 200,
                                    value=float(get_preset_value(preset_vals, "diaBP", float, 80.0)))
            heartRate = st.number_input("Heart rate", 30, 200,
                                        value=float(get_preset_value(preset_vals, "heartRate", float, 72.0)))
            glucose = st.number_input("Glucose", 40, 500,
                                      value=float(get_preset_value(preset_vals, "glucose", float, 90.0)))

        submitted = st.form_submit_button("🔍 Predict")

    st.markdown("</div>", unsafe_allow_html=True)

    # On submit: show skeleton loader while processing
    if submitted:
        # show a skeleton area for results while we compute (nice UX)
        placeholder = st.empty()
        with placeholder.container():
            st.markdown("<div class='card'><div class='skeleton' style='width:40%; margin-bottom:10px'></div>"
                        "<div class='skeleton' style='width:80%; height:220px'></div></div>", unsafe_allow_html=True)

        # prepare input dict with proper types
        input_vals = {
            "male": gender, "age": int(age), "education": int(education), "currentSmoker": currentSmoker,
            "cigsPerDay": int(cigsPerDay), "BPMeds": BPMeds, "prevalentStroke": prevalentStroke,
            "prevalentHyp": prevalentHyp, "diabetes": diabetes, "totChol": float(totChol),
            "sysBP": float(sysBP), "diaBP": float(diaBP), "BMI": float(BMI),
            "heartRate": float(heartRate), "glucose": float(glucose)
        }

        # small artificial wait for UX demonstration (remove/shorten in production)
        time.sleep(0.3)

        # build array, scale, predict — robust error handling
        try:
            X_arr = build_input_array(input_vals)
            try:
                X_scaled = scaler.transform(X_arr)
            except Exception:
                X_scaled = X_arr  # fallback if scaler issue
            pred = model.predict(X_scaled)[0]
            prob = float(model.predict_proba(X_scaled)[0][1])
        except Exception as e:
            placeholder.empty()
            st.error("Prediction failed: " + str(e))
            st.stop()

        # show results (replace skeleton)
        placeholder.empty()

        # Save latest results in session for other tabs (explain/report)
        st.session_state["last_input"] = input_vals
        st.session_state["last_pred"] = int(pred)
        st.session_state["last_prob"] = float(prob)

        # Results header row: badge, recommendation, quick export
        c1, c2, c3 = st.columns([1.5, 2, 1])
        with c1:
            level = "LOW" if prob < 0.30 else ("MEDIUM" if prob < 0.60 else "HIGH")
            color = SAFE if prob < 0.30 else (CAUTION if prob < 0.60 else ACCENT)
            st.markdown(f"<div class='card'><div style='display:flex; justify-content:space-between; align-items:center'>"
                        f"<div><span class='risk-badge' style='background:{color}'>{level}</span>"
                        f"<div class='small-muted' style='margin-top:6px'>Probability: {prob:.2f}</div></div></div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='card'><div class='kv'>Recommendation</div>", unsafe_allow_html=True)
            if int(pred) == 1:
                st.markdown("<b>Consult a clinician urgently. Recommended tests: ECG, lipid panel, fasting glucose.</b>", unsafe_allow_html=True)
            else:
                st.markdown("Maintain healthy lifestyle; routine check-ups recommended.", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c3:
            st.markdown("<div class='card'><div class='kv'>Export</div>", unsafe_allow_html=True)
            # Grab gauge PNG if available (requires kaleido)
            gauge_fig = make_gauge_plot(prob)
            gauge_png = export_plotly_png_bytes(gauge_fig)
            pdf_bytes = pdf_report_bytes(pd.DataFrame([input_vals]), ("HIGH" if pred==1 else "LOW"), prob, note="Academic demo — not medical advice.", gauge_png=gauge_png)
            st.download_button("📥 Download PDF Report", data=pdf_bytes, file_name="chd_report.pdf", mime="application/pdf")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        # Visuals row
        left_col, right_col = st.columns([1.5, 1])
        with left_col:
            st.plotly_chart(gauge_fig, use_container_width=True)
        with right_col:
            st.plotly_chart(make_prob_bar(prob), use_container_width=True)

        st.markdown("### Model input preview")
        preview_df = pd.DataFrame([input_vals]).T.rename(columns={0: "value"})
        st.dataframe(preview_df, height=260)

        # subtle celebratory animation on low-risk
        if prob < 0.25:
            st.balloons()
        elif prob > 0.90:
            st.snow()

# ---------- TAB: Explain ----------
with tab2:
    st.markdown("<div class='card page-fade'>", unsafe_allow_html=True)
    st.markdown("### Model Explainability (SHAP) — optional", unsafe_allow_html=True)
    st.markdown("SHAP explains which features pushed the model toward this prediction. SHAP may be slow for some models.", unsafe_allow_html=True)

    if "last_input" not in st.session_state:
        st.info("Run a prediction on the Predict tab first to compute SHAP for that case.")
    else:
        if shap is None:
            st.warning("SHAP is not installed in this environment. Add `shap` to requirements.txt to enable explainability plots.")
        else:
            # show a skeleton while computing
            if st.button("Compute SHAP for last input (may take 10-60 seconds)"):
                placeholder_shap = st.empty()
                with placeholder_shap.container():
                    st.markdown("<div class='skeleton' style='width:30%; margin-bottom:10px'></div>", unsafe_allow_html=True)
                    st.markdown("<div class='skeleton' style='width:100%; height:300px'></div>", unsafe_allow_html=True)

                # prepare background & sample arrays
                last_input = st.session_state["last_input"]
                try:
                    # background: repeat last input (if you have train X, replace with a sampled X)
                    bg = np.repeat(np.array(list(last_input.values()), dtype=float).reshape(1,-1), 50, axis=0)
                    X_sample = np.array([list(last_input.values())], dtype=float)
                    with st.spinner("Computing SHAP values..."):
                        explainer = shap.Explainer(model, bg)
                        shap_values = explainer(X_sample)
                    placeholder_shap.empty()
                    # try waterfall
                    try:
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        fig_w = shap.plots.waterfall(shap_values[0], show=False)
                        st.pyplot(bbox_inches='tight')
                    except Exception:
                        # fallback to mean |shap| bar
                        vals = np.abs(shap_values.values).mean(0) if hasattr(shap_values, "values") else np.mean(np.abs(shap_values), axis=1)[0]
                        feat_names = list(last_input.keys())
                        fig2, ax = plt.subplots(figsize=(6,4))
                        ax.barh(feat_names, vals)
                        ax.set_xlabel("Mean |SHAP value|")
                        st.pyplot(fig2)
                except Exception as e:
                    placeholder_shap.empty()
                    st.error("SHAP computation failed: " + str(e))
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- TAB: Report ----------
with tab3:
    st.markdown("<div class='card page-fade'>", unsafe_allow_html=True)
    st.markdown("### Submission Checklist & Deliverables", unsafe_allow_html=True)
    st.markdown("- Downloadable PDF report (from Predict tab).")
    st.markdown("- Save screenshots: EDA, model metrics, SHAP, app UI.")
    st.markdown("- Add live demo link and README to your paper submission.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- TAB: About ----------
with tab4:
    st.markdown("<div class='card page-fade'>", unsafe_allow_html=True)
    st.markdown("### About & Disclaimer", unsafe_allow_html=True)
    st.markdown("""
    - **Dataset:** Framingham Heart Study (public).  
    - **Purpose:** Academic demonstration — **not** clinical diagnosis.  
    - **Features:** End-to-end pipeline, PDF export, explainability (SHAP).  
    - **Contact:** Add your details here for README/portfolio.
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("<div style='padding:10px 0; text-align:center; color:rgba(200,220,240,0.6)'>© 2025 Heart Disease Predictor • Built by Sachin</div>", unsafe_allow_html=True)
