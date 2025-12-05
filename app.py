# app.py — PREMIUM UI/UX (drop-in)
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
from fpdf import FPDF

# -------------------------
# CONFIG & ASSETS
# -------------------------
st.set_page_config(page_title="Heart Disease Predictor — Premium", page_icon="❤️", layout="wide")
# If you have a logo file named 'logo.png' in repo root, it will appear. Optional.
LOGO_PATH = "logo.png"  # add to repo root if you want brand logo

# Load model + scaler (your filenames from repo)
model = joblib.load("best_heart_chd_model.joblib")
scaler = joblib.load("scaler_chd.joblib")

# -------------------------
# STYLES: modern gradient + cards
# -------------------------
PRIMARY = "#0f1724"    # deep background
ACCENT = "#ff4b4b"     # accent red
CARD_BG = "#0b1220"
TEXT = "#e6eef8"

css = f"""
<style>
:root {{
  --bg: {PRIMARY};
  --card: {CARD_BG};
  --accent: {ACCENT};
  --text: {TEXT};
}}
body {{
  background: linear-gradient(180deg,#071223 0%, #0b1220 100%);
  color: var(--text);
}}
.stApp > header {{display: none;}}
.logo {{
  display:flex; align-items:center; gap:12px;
}}
.card {{
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  padding: 18px; border-radius: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.6);
  border: 1px solid rgba(255,255,255,0.03);
}}
.small-muted {{ color: #9fb0c9; font-size:13px; }}
.kv {{ font-weight:600; color:#cfe7ff; }}
.btn-primary {{ background: linear-gradient(90deg,#ff6b6b,#ff4b4b); padding:10px 18px; border-radius:8px; color:white; }}
.footer {{ color:#7b8ba3; font-size:12px; text-align:center; padding-top:18px; }}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# -------------------------
# Utilities
# -------------------------
def to_bin(x):
    return 1 if x in ("Yes", "Male") else 0

def build_input_array(vals):
    # ensure same order as training
    arr = np.array([[
        to_bin(vals['male']), vals['age'], vals['education'],
        to_bin(vals['currentSmoker']), vals['cigsPerDay'],
        to_bin(vals['BPMeds']), to_bin(vals.get('prevalentStroke', 0)),
        to_bin(vals.get('prevalentHyp', 0)), to_bin(vals['diabetes']),
        vals['totChol'], vals['sysBP'], vals['diaBP'],
        vals['BMI'], vals.get('heartRate', 72), vals['glucose']
    ]], dtype=float)
    return arr

def make_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        domain={'x':[0,1],'y':[0,1]},
        title={'text': "Risk %"},
        gauge={'axis':{'range':[0,100]},
               'bar':{'color': ACCENT},
               'steps':[{'range':[0,30],'color':'#16a34a'},
                        {'range':[30,60],'color':'#f59e0b'},
                        {'range':[60,100],'color':'#ef4444'}]}
    ))
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def make_prob_chart(prob):
    fig = go.Figure(go.Bar(x=["CHD Risk"], y=[prob], marker_color=[ACCENT]))
    fig.update_yaxes(range=[0,1], tickformat=".0%")
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=220, paper_bgcolor="rgba(0,0,0,0)")
    return fig

def pdf_bytes_from_report(input_df, label, prob, note="", gauge_png=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 8, "Heart Disease Risk Report", ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Arial", size=11)
    pdf.cell(0,6, f"Prediction: {label}", ln=True)
    pdf.cell(0,6, f"Risk probability: {prob:.2f}", ln=True)
    pdf.ln(6)
    if gauge_png:
        # gauge_png should be bytes PNG
        with open("tmp_gauge.png","wb") as f:
            f.write(gauge_png)
        pdf.image("tmp_gauge.png", x=30, w=150)
        pdf.ln(4)
    pdf.set_font("Arial", size=10)
    pdf.ln(2)
    pdf.cell(0,6,"Inputs:", ln=True)
    for k,v in input_df.iloc[0].items():
        pdf.cell(0,5, f" - {k}: {v}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# -------------------------
# HERO
# -------------------------
colh1, colh2 = st.columns([1,2])
with colh1:
    if st.sidebar.checkbox("Show brand header", value=True):
        try:
            st.image(LOGO_PATH, width=80)
        except:
            st.markdown("<div class='logo'><h2 style='color:#ff6b6b;'>❤️ CHD Predictor</h2></div>", unsafe_allow_html=True)
with colh2:
    st.markdown("<h2 style='text-align:left; color:#cfe7ff;'>Professional Heart Disease Risk Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Premium UI • Exportable reports • Explainable AI</div>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------
# TABS: Predict / Explain / Report / About
# -------------------------
tabs = st.tabs(["Predict", "Explainability", "Report", "About"])

# ---------- TAB 1: Predict ----------
with tabs[0]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Patient details (quick presets available)", unsafe_allow_html=True)

    # Preset examples to quickly populate for demo
    presets = {
        "Sample: Healthy (low risk)": {
            'male':'Male','age':45,'education':2,'currentSmoker':'No','cigsPerDay':0,'BPMeds':'No',
            'prevalentStroke':0,'prevalentHyp':0,'diabetes':'No','totChol':180,'sysBP':120,'diaBP':78,'BMI':23,'heartRate':72,'glucose':85
        },
        "Sample: High risk smoker (demo)": {
            'male':'Male','age':62,'education':1,'currentSmoker':'Yes','cigsPerDay':20,'BPMeds':'Yes',
            'prevalentStroke':0,'prevalentHyp':1,'diabetes':'Yes','totChol':260,'sysBP':150,'diaBP':90,'BMI':30,'heartRate':75,'glucose':140
        }
    }

    preset_choice = st.selectbox("Choose example to load", options=["Custom"] + list(presets.keys()))
    vals = None
    if preset_choice != "Custom":
        vals = presets[preset_choice]

    # Input form arranged in columns
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 20, 100, value=vals['age'] if vals else 50)
            male = st.selectbox("Gender", ["Male","Female"], index=0 if (vals and vals['male']=='Male') else 1)
            education = st.number_input("Education (1–4)", 1, 4, value=vals['education'] if vals else 1)
            currentSmoker = st.selectbox("Current Smoker?", ["Yes","No"], index=0 if (vals and vals['currentSmoker']=='Yes') else 1)
            cigsPerDay = st.number_input("Cigarettes/day", 0, 100, value=vals['cigsPerDay'] if vals else 0)
        with c2:
            BPMeds = st.selectbox("On BP medication?", ["Yes","No"], index=0 if (vals and vals['BPMeds']=='Yes') else 1)
            prevalentStroke = st.selectbox("Stroke history?", ["Yes","No"], index=0)
            prevalentHyp = st.selectbox("Hypertension?", ["Yes","No"], index=1 if (vals and vals.get('prevalentHyp',0)) else 0)
            diabetes = st.selectbox("Diabetes?", ["Yes","No"], index=0 if (vals and vals['diabetes']=='Yes') else 1)
            BMI = st.number_input("BMI", 10.0, 60.0, value=vals['BMI'] if vals else 25.0)
        with c3:
            totChol = st.number_input("Total cholesterol", 100, 600, value=vals['totChol'] if vals else 200)
            sysBP = st.number_input("Systolic BP", 80, 250, value=vals['sysBP'] if vals else 120)
            diaBP = st.number_input("Diastolic BP", 40, 150, value=vals['diaBP'] if vals else 80)
            heartRate = st.number_input("Heart rate", 40, 200, value=vals['heartRate'] if vals else 72)
            glucose = st.number_input("Glucose", 40, 300, value=vals['glucose'] if vals else 90)

        submitted = st.form_submit_button("🔍 Predict")

    st.markdown("</div>", unsafe_allow_html=True)

    # On submit -> predict and show premium result cards
    if submitted:
        input_vals = {
            'male':male, 'age':age, 'education':education, 'currentSmoker':currentSmoker,
            'cigsPerDay':cigsPerDay, 'BPMeds':BPMeds, 'prevalentStroke':prevalentStroke,
            'prevalentHyp':prevalentHyp, 'diabetes':diabetes, 'totChol':totChol,
            'sysBP':sysBP, 'diaBP':diaBP, 'BMI':BMI, 'heartRate':heartRate, 'glucose':glucose
        }
        x = build_input_array(input_vals)
        x_scaled = scaler.transform(x)
        pred = model.predict(x_scaled)[0]
        prob = model.predict_proba(x_scaled)[0][1]

        # top summary row: risk badge + quick actions
        rcol1, rcol2, rcol3 = st.columns([1.2,1,1])
        with rcol1:
            color = "#16a34a" if prob < 0.3 else ("#f59e0b" if prob < 0.6 else "#ef4444")
            level = "LOW" if prob < 0.3 else ("MEDIUM" if prob < 0.6 else "HIGH")
            st.markdown(f"<div class='card'><h3 style='color:{color};'>Risk: <span style='font-size:28px'>{level}</span></h3><div class='small-muted'>Probability: {prob:.2f}</div></div>", unsafe_allow_html=True)
        with rcol2:
            st.markdown("<div class='card'><div class='kv'>Recommendation</div><div class='small-muted'>", unsafe_allow_html=True)
            if pred == 1:
                st.markdown("<b>Consult a doctor urgently. Consider tests: ECG, lipid profile, blood sugar.</b>", unsafe_allow_html=True)
            else:
                st.markdown("Continue healthy lifestyle; annual checkups recommended.", unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)
        with rcol3:
            # Quick export buttons
            st.markdown("<div class='card'><div class='kv'>Export</div>", unsafe_allow_html=True)
            # Gauge image bytes for PDF
            gauge_fig = make_gauge(prob)
            try:
                png_bytes = gauge_fig.to_image(format="png", width=600, height=350, scale=2)
            except Exception:
                png_bytes = None
            # create small inline pdf
            pdf_bytes = pdf_bytes_from_report(pd.DataFrame([input_vals]), "HIGH" if pred==1 else "LOW", prob, note="Academic demo", gauge_png=png_bytes)
            st.download_button("📥 Download PDF", data=pdf_bytes, file_name="chd_report.pdf", mime="application/pdf")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        # Visualizations row
        g1, g2 = st.columns([1,1.2])
        with g1:
            st.plotly_chart(make_gauge(prob), use_container_width=True)
        with g2:
            st.plotly_chart(make_prob_chart(prob), use_container_width=True)

        # Simple feature preview (for explanation)
        st.markdown("### Model input preview")
        inp_df = pd.DataFrame([input_vals])
        st.dataframe(inp_df.T.style.set_properties(**{'background-color':'#071425','color':'#cfe7ff'}), height=300)

        # Place interactive advise card
        st.markdown("<div class='card'><b>Teacher Note:</b> Save this link and attach the PDF + screenshots for your report submission.</div>", unsafe_allow_html=True)

        # Store latest values in session for other tabs
        st.session_state["last_input"] = input_vals
        st.session_state["last_pred"] = int(pred)
        st.session_state["last_prob"] = float(prob)

# ---------- TAB 2: Explainability ----------
with tabs[1]:
    st.markdown("## Model Explainability")
    st.markdown("Use SHAP to inspect which features pushed the model towards the final prediction. (May take a short time.)")
    if "last_input" not in st.session_state:
        st.info("Run a prediction on the 'Predict' tab first to generate explainability for that input.")
    else:
        if st.button("Compute SHAP (may take 10-60s)"):
            import shap
            # prepare background sample (we try to sample from training X if available)
            try:
                background = pd.DataFrame([st.session_state["last_input"]])
                bg_arr = np.repeat(background.values, 30, axis=0)
            except Exception:
                bg_arr = np.array([list(st.session_state["last_input"].values())])
            # compute (Tree explainer for tree models; fallback to Kernel)
            try:
                explainer = shap.Explainer(model, bg_arr)
                sv = explainer(np.array([list(st.session_state["last_input"].values())]))
                # waterfall plot if available
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot(shap.plots.waterfall(sv[0], show=False))
            except Exception as e:
                st.error("SHAP computation failed or took too long: " + str(e))

# ---------- TAB 3: Report ----------
with tabs[2]:
    st.markdown("## Report & Notes")
    st.markdown("You can download the PDF report from the Predict tab. Include it in your submission. Below are helpful items for your IEEE paper and LinkedIn.")
    st.markdown("### Quick checklist")
    st.markdown("- Screenshots: EDA, Model metrics, Feature importance, App UI\n- Include link to live app (Streamlit).")

# ---------- TAB 4: About ----------
with tabs[3]:
    st.markdown("## About this project")
    st.markdown("""
    **Dataset:** Framingham Heart Study (public)  
    **Models tested:** Logistic, Tree, RandomForest, GradientBoosting, MLP  
    **Final model:** exported and used in this app  
    """)
    st.markdown("<div class='footer'>© 2025 Heart Disease Predictor • Built by Sachin</div>", unsafe_allow_html=True)
