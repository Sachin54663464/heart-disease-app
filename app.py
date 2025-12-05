# app.py — Final premium Clinical CHD Dashboard (fixed)
# Expects model files in repo root:
#   - best_heart_chd_model.joblib
#   - scaler_chd.joblib
#
# Add heavy libs to requirements.txt if you want SHAP/kaleido functionality.

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
from fpdf import FPDF
import io
import matplotlib.pyplot as plt

# SHAP is optional — app will warn if it's not installed
try:
    import shap
except Exception:
    shap = None

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="Clinical CHD Dashboard (Premium)", page_icon="❤️", layout="wide")
MODEL_FILENAME = "best_heart_chd_model.joblib"
SCALER_FILENAME = "scaler_chd.joblib"

# ---------------------------
# Load model + scaler (safe)
# ---------------------------
try:
    model = joblib.load(MODEL_FILENAME)
    scaler = joblib.load(SCALER_FILENAME)
except Exception as e:
    st.error(f"Model or scaler not found or failed to load. Make sure files exist in repo root:\n- {MODEL_FILENAME}\n- {SCALER_FILENAME}\n\nError: {e}")
    st.stop()

# ---------------------------
# Styles (Apple-ish premium)
# ---------------------------
st.markdown(
    """
    <style>
      :root{
        --accent: #ff4b4b;
        --accent2: #ff6b6b;
        --safe: #16a34a;
        --caution: #f59e0b;
        --dark: #0b1220;
        --muted: #9fb0c9;
        --card: rgba(255,255,255,0.03);
      }
      body { background: linear-gradient(180deg,#071223 0%, var(--dark) 100%); color: #e6eef8; }
      .header-title { font-size:34px; font-weight:800; margin-bottom:4px; text-align:center; }
      .header-sub { color:var(--muted); margin-top:-8px; margin-bottom: 18px; text-align:center; }
      .card { background:var(--card); padding:18px; border-radius:14px; border:1px solid rgba(255,255,255,0.03); box-shadow:0 8px 30px rgba(2,6,23,0.6); animation:fadeUp .4s ease both; }
      @keyframes fadeUp { from {opacity:0; transform:translateY(6px)} to {opacity:1; transform:none} }
      .skeleton { height:14px; border-radius:8px; background: linear-gradient(90deg, rgba(255,255,255,0.03), rgba(255,255,255,0.06), rgba(255,255,255,0.03)); background-size:200% 100%; animation:shimmer 1.2s infinite; margin-bottom:8px; }
      @keyframes shimmer { 0% {background-position:200% 0} 100% {background-position:-200% 0} }
      .muted { color:var(--muted); font-size:13px; }
      .kv { font-weight:700; color:#dff2ff; }
      .risk-badge { padding:8px 12px; border-radius:12px; color:white; font-weight:700; display:inline-block; }
      .btn { background: linear-gradient(90deg,var(--accent2),var(--accent)); color:white; padding:8px 14px; border-radius:10px; font-weight:700; border:none; }
      .small { font-size:13px; color:var(--muted); }
      @media (max-width:768px){ .header-title { font-size:22px; } }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Utilities
# ---------------------------
def to_bin(x):
    return 1 if str(x).lower() in ("yes", "male", "1", "true") else 0

def safe_cast(value, dtype, default):
    try:
        if dtype == int:
            return int(float(value))
        if dtype == float:
            return float(value)
        return str(value)
    except Exception:
        return default

def get_preset_value(presets, key, dtype, default):
    if not presets:
        return default
    raw = presets.get(key, default)
    return safe_cast(raw, dtype, default)

def build_input_array(d):
    arr = np.array([[
        to_bin(d["male"]),
        float(d["age"]),
        float(d["education"]),
        to_bin(d["currentSmoker"]),
        float(d["cigsPerDay"]),
        to_bin(d["BPMeds"]),
        to_bin(d.get("prevalentStroke", 0)),
        to_bin(d.get("prevalentHyp", 0)),
        to_bin(d["diabetes"]),
        float(d["totChol"]),
        float(d["sysBP"]),
        float(d["diaBP"]),
        float(d["BMI"]),
        float(d.get("heartRate", 72)),
        float(d["glucose"])
    ]], dtype=float)
    return arr

def make_gauge_figure(prob):
    val = float(prob) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number={'suffix':'%','font':{'size':36}},
        title={'text':"10-year CHD Risk"},
        gauge={'axis':{'range':[0,100]},
               'steps':[{'range':[0,30],'color':'#16a34a'},{'range':[30,60],'color':'#f59e0b'},{'range':[60,100],'color':'#ff4b4b'}],
               'bar':{'color':'#ff4b4b'}}
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=420, margin=dict(t=20,b=10,l=10,r=10))
    return fig

def make_prob_bar(prob):
    fig = go.Figure(go.Bar(x=["CHD risk"], y=[prob], marker_color="#ff4b4b"))
    fig.update_layout(yaxis=dict(range=[0,1], tickformat=".0%"), paper_bgcolor="rgba(0,0,0,0)", height=220, margin=dict(t=10,b=10))
    return fig

def export_fig_png(fig):
    try:
        img = fig.to_image(format="png", width=900, height=480, scale=2)
        return img
    except Exception:
        return None

def pdf_from_input(input_df, pred_label, prob, gauge_png=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 8, "Clinical Heart Disease Risk Report", ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 6, f"Prediction: {pred_label}", ln=True)
    pdf.cell(0, 6, f"Probability: {prob:.2f}", ln=True)
    pdf.ln(6)
    if gauge_png:
        try:
            tmp = "/tmp/gauge.png"
            with open(tmp, "wb") as f:
                f.write(gauge_png)
            pdf.image(tmp, x=25, w=160)
            pdf.ln(6)
        except Exception:
            pass
    pdf.set_font("Arial", size=10)
    pdf.cell(0,6, "Input values:", ln=True)
    for k,v in input_df.iloc[0].items():
        pdf.multi_cell(0,5, f" - {k}: {v}")
    return pdf.output(dest="S").encode("latin-1")

# ---------------------------
# Presets
# ---------------------------
PRESETS = {
    "Healthy (demo)": {
        "male": "Female","age":38,"education":3,"currentSmoker":"No","cigsPerDay":0,"BPMeds":"No","prevalentStroke":"No","prevalentHyp":"No","diabetes":"No",
        "totChol":180,"sysBP":115,"diaBP":75,"BMI":22.1,"heartRate":68,"glucose":85
    },
    "High-risk smoker (demo)": {
        "male":"Male","age":58,"education":1,"currentSmoker":"Yes","cigsPerDay":20,"BPMeds":"Yes","prevalentStroke":"No","prevalentHyp":"Yes","diabetes":"Yes",
        "totChol":250,"sysBP":160,"diaBP":95,"BMI":29.5,"heartRate":90,"glucose":140
    }
}

# ---------------------------
# Header
# ---------------------------
st.markdown("<div class='header-title'>Clinical Heart Disease Risk Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='header-sub'>Premium clinical UI • Exportable PDF reports • Explainability (SHAP)</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.markdown("## ❤️ CHD Predictor")
    st.markdown("Premium clinical-style dashboard")
    st.markdown("---")
    preset_choice = st.selectbox("Load preset", ["Custom"] + list(PRESETS.keys()))
    if st.button("Clear preset"):
        preset_choice = "Custom"
    st.markdown("---")
    st.markdown("### Developer")
    st.markdown("Built by Sachin — academic demo (not clinical).")

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Explain (SHAP)", "Report", "About"])

# ---------------------------
# Predict tab
# ---------------------------
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Patient details — enter or load a preset", unsafe_allow_html=True)

    preset_vals = PRESETS.get(preset_choice) if preset_choice in PRESETS else None

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 18, 120, value=get_preset_value(preset_vals, "age", int, 50))
            male = st.selectbox("Gender", ["Female","Male"], index=0 if get_preset_value(preset_vals, "male", str, "Female") == "Female" else 1)
            education = st.number_input("Education (1–4)", 1, 4, value=get_preset_value(preset_vals, "education", int, 1))
            currentSmoker = st.selectbox("Current smoker?", ["No","Yes"], index=0 if get_preset_value(preset_vals, "currentSmoker", str, "No")=="No" else 1)
            cigsPerDay = st.number_input("Cigarettes / day", 0, 200, value=get_preset_value(preset_vals, "cigsPerDay", int, 0))
        with c2:
            BPMeds = st.selectbox("On BP medication?", ["No","Yes"], index=0 if get_preset_value(preset_vals, "BPMeds", str, "No")=="No" else 1)
            prevalentStroke = st.selectbox("Stroke history?", ["No","Yes"], index=0 if get_preset_value(preset_vals, "prevalentStroke", str, "No")=="No" else 1)
            prevalentHyp = st.selectbox("Hypertension?", ["No","Yes"], index=0 if get_preset_value(preset_vals, "prevalentHyp", str, "No")=="No" else 1)
            diabetes = st.selectbox("Diabetes?", ["No","Yes"], index=0 if get_preset_value(preset_vals, "diabetes", str, "No")=="No" else 1)
            BMI = st.number_input("BMI", 10.0, 60.0, value=float(get_preset_value(preset_vals, "BMI", float, 25.0)))
        with c3:
            totChol = st.number_input("Total cholesterol", 100.0, 600.0, value=float(get_preset_value(preset_vals, "totChol", float, 200.0)))
            sysBP = st.number_input("Systolic BP", 80.0, 250.0, value=float(get_preset_value(preset_vals, "sysBP", float, 120.0)))
            diaBP = st.number_input("Diastolic BP", 40.0, 150.0, value=float(get_preset_value(preset_vals, "diaBP", float, 80.0)))
            heartRate = st.number_input("Heart Rate", 30.0, 200.0, value=float(get_preset_value(preset_vals, "heartRate", float, 72.0)))
            glucose = st.number_input("Glucose", 40.0, 400.0, value=float(get_preset_value(preset_vals, "glucose", float, 90.0)))

        submit = st.form_submit_button("🔍 Predict Risk")

    st.markdown("</div>", unsafe_allow_html=True)

    if submit:
        placeholder = st.empty()
        with placeholder.container():
            st.markdown("<div class='card'><div class='skeleton' style='width:28%'></div><div class='skeleton' style='width:100%; height:220px'></div></div>", unsafe_allow_html=True)
            time.sleep(0.25)

        input_vals = {
            "male": male, "age": int(age), "education": int(education), "currentSmoker": currentSmoker,
            "cigsPerDay": int(cigsPerDay), "BPMeds": BPMeds, "prevalentStroke": prevalentStroke,
            "prevalentHyp": prevalentHyp, "diabetes": diabetes, "totChol": float(totChol),
            "sysBP": float(sysBP), "diaBP": float(diaBP), "BMI": float(BMI),
            "heartRate": float(heartRate), "glucose": float(glucose)
        }

        try:
            X = build_input_array(input_vals)
            try:
                Xs = scaler.transform(X)
            except Exception:
                Xs = X
            pred = model.predict(Xs)[0]
            prob = float(model.predict_proba(Xs)[0][1])
        except Exception as e:
            placeholder.empty()
            st.error("Prediction failed: " + str(e))
            st.stop()

        st.session_state["last_input"] = input_vals
        st.session_state["last_pred"] = int(pred)
        st.session_state["last_prob"] = float(prob)

        placeholder.empty()

        colA, colB, colC = st.columns([1.4, 2, 1])
        with colA:
            level = "LOW" if prob < 0.30 else ("MEDIUM" if prob < 0.60 else "HIGH")
            color = "#16a34a" if prob < 0.30 else ("#f59e0b" if prob < 0.60 else "#ff4b4b")
            st.markdown(f"<div class='card'><div style='display:flex;align-items:center;justify-content:space-between'><div><span style='font-size:14px;color:{color};font-weight:800'>Risk: {level}</span><div class='muted'>Probability: {prob:.2f}</div></div></div></div>", unsafe_allow_html=True)
        with colB:
            st.markdown("<div class='card'><div class='kv'>Recommendation</div>", unsafe_allow_html=True)
            if int(pred)==1:
                st.markdown("<b>Consult a clinician urgently. Consider ECG, lipid panel, fasting glucose.</b>", unsafe_allow_html=True)
            else:
                st.markdown("Maintain a healthy lifestyle & monitor vitals regularly.", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with colC:
            st.markdown("<div class='card'><div class='kv'>Export</div>", unsafe_allow_html=True)
            gfig = make_gauge_figure(prob)
            gpng = export_fig_png(gfig)
            pdf_bytes = pdf_from_input(pd.DataFrame([input_vals]), ("HIGH" if pred==1 else "LOW"), prob, gauge_png=gpng)
            st.download_button("📥 Download PDF Report", data=pdf_bytes, file_name="chd_report.pdf", mime="application/pdf")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        left, right = st.columns([1.5,1])
        with left:
            st.plotly_chart(gfig, use_container_width=True)
        with right:
            st.plotly_chart(make_prob_bar(prob), use_container_width=True)

        st.markdown("### Model input preview")
        df_preview = pd.DataFrame([input_vals]).T.rename(columns={0:"value"})
        st.dataframe(df_preview, height=260)

        if prob < 0.25:
            st.balloons()
        elif prob > 0.9:
            st.snow()

# ---------------------------
# Explain (SHAP) tab
# ---------------------------
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### SHAP Explainability — per-case feature contributions", unsafe_allow_html=True)
    st.markdown("Run SHAP to see which features pushed this prediction higher or lower. (SHAP can be slow for non-tree models.)", unsafe_allow_html=True)

    if "last_input" not in st.session_state:
        st.info("Run a prediction on the Predict tab first to compute SHAP for that case.")
    else:
        if shap is None:
            st.warning("SHAP is not installed. Add `shap` to requirements.txt to enable explainability.")
        else:
            if st.button("Compute SHAP for last case (may take time)"):
                placeholder_shap = st.empty()
                with placeholder_shap.container():
                    st.markdown("<div class='skeleton' style='width:30%'></div><div class='skeleton' style='width:100%; height:300px'></div>", unsafe_allow_html=True)
                    time.sleep(0.25)
                last_input = st.session_state["last_input"]
                try:
                    X_case = build_input_array(last_input)
                    background = np.repeat(X_case, 50, axis=0)
                    with st.spinner("Computing SHAP values..."):
                        try:
                            explainer = shap.TreeExplainer(model)
                            # shap_values shape depends on model type; we handle generically below
                            shap_values = explainer.shap_values(X_case) if hasattr(explainer, "shap_values") else explainer(X_case)
                        except Exception:
                            explainer = shap.Explainer(model.predict_proba, background)
                            shap_values = explainer(X_case)
                    placeholder_shap.empty()
                    # Try waterfall or fallback bar
                    try:
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        shap.plots.waterfall(shap_values[0] if isinstance(shap_values, (list,tuple)) else shap_values[0], show=True)
                        st.pyplot()
                    except Exception:
                        vals = shap_values.values if hasattr(shap_values, "values") else (shap_values[0] if isinstance(shap_values, (list,tuple)) else shap_values)
                        abs_mean = np.abs(vals).mean(0)
                        feat_names = list(last_input.keys())
                        fig2, ax = plt.subplots(figsize=(6,4))
                        ax.barh(feat_names, abs_mean)
                        ax.set_xlabel("Mean |SHAP value|")
                        st.pyplot(fig2)
                except Exception as e:
                    placeholder_shap.empty()
                    st.error("SHAP computation failed: " + str(e))
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Report & About tabs
# ---------------------------
with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Submission checklist & exports", unsafe_allow_html=True)
    st.markdown("- Download the PDF report (Predict tab).")
    st.markdown("- Save EDA/model metrics and SHAP plots for your paper.")
    st.markdown("- Include live demo and README in your submission.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### About & Disclaimer", unsafe_allow_html=True)
    st.markdown("This is an academic demo using the Framingham dataset and a trained classifier. It is NOT medical advice.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown("<div style='text-align:center; color:rgba(200,220,240,0.6); padding:14px 0;'>Built by Sachin • Academic demo only</div>", unsafe_allow_html=True)
