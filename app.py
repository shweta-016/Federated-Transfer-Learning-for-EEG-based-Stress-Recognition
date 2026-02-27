import streamlit as st
import torch
import numpy as np
import os
import time

from model_eegnet import EEGNet
from normalization import z_score_normalize
#from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="EEG Stress Detection",
    layout="wide",
    page_icon="🧠"
)

# --------------------------------------------------
# HIGH-CONTRAST DARK CSS (ALL TEXT VISIBLE)
st.markdown("""
<style>

/* ===== GLOBAL TEXT ===== */
html, body, .stApp, p, span, div, label {
    color: #ffffff !important;
    font-family: 'Segoe UI', sans-serif;
}

/* BACKGROUND */
.stApp {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
}

/* HEADINGS */
h1,h2,h3,h4,h5,h6 {
    color:#ffffff !important;
}

/* RADIO OPTIONS (ALL STATES) */
div[role="radiogroup"] label span {
    color:#ffffff !important;
    opacity:1 !important;
    font-weight:600;
}

/* SELECTED RADIO */
div[role="radiogroup"] input:checked + div span {
    color:#00ffaa !important;
}

/* INFO / WARNING / SUCCESS / ERROR TEXT */
.stInfo, .stWarning, .stSuccess, .stError {
    color:#ffffff !important;
    font-weight:600;
}

/* INFO BOX BACKGROUND */
.stInfo {background:rgba(0,150,255,0.18) !important;}
.stWarning {background:rgba(255,200,0,0.18) !important;}
.stSuccess {background:rgba(0,255,170,0.18) !important;}
.stError {background:rgba(255,80,80,0.18) !important;}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.08);
    border-radius:12px;
    padding:18px;
}

/* DROPZONE TEXT */
[data-testid="stFileUploaderDropzone"] {
    background:#e6eef2 !important;
}
[data-testid="stFileUploaderDropzone"] * {
    color:#0f2027 !important;
    font-weight:600;
}

/* BUTTON */
button[kind="secondary"] {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color:white !important;
    border-radius:8px;
}

/* METRICS */
[data-testid="stMetricLabel"] {color:#cccccc !important;}
[data-testid="stMetricValue"] {color:#00ffaa !important;}

/* PROGRESS BAR */
div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg,#ff512f,#dd2476);
}

/* GLASS PANEL */
.glass-container {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    border-radius:16px;
    padding:20px;
    margin-top:15px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.markdown("<h1>🧠 EEG Stress Detection System</h1>", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sample_folder = "normalized_epochs/subject_01"
    files = [f for f in os.listdir(sample_folder) if f.endswith(".npy")]
    sample = np.load(os.path.join(sample_folder, files[0]))

    ch, smp = sample.shape

    model = EEGNet(num_channels=ch, samples=smp, num_classes=2)
    model.load_state_dict(
        torch.load("server_saved_models/final_finetuned.pth", map_location=device)
    )
    model.to(device)
    model.eval()

    return model, device, ch, smp

model, device, channels, samples = load_model()

# --------------------------------------------------
# INPUT MODE
# --------------------------------------------------
st.markdown("## 📥 Choose Input Method")

input_mode = st.radio(
    "",
    ["Upload EEG file","Use demo EEG sample"],
    horizontal=True
)

data = None

# --------------------------------------------------
# OPTION 1: UPLOAD
# --------------------------------------------------
if input_mode == "Upload EEG file":
    file = st.file_uploader("📂 Upload EEG (.npy)", type=["npy"])
    if file:
        st.success("EEG file loaded")
        data = np.load(file)

# --------------------------------------------------
# OPTION 2: DEMO
# --------------------------------------------------
elif input_mode == "Use demo EEG sample":
    st.info("Using built-in EEG sample")
    folder = "normalized_epochs/subject_01"
    f = [x for x in os.listdir(folder) if x.endswith(".npy")]
    data = np.load(os.path.join(folder, f[0]))


# --------------------------------------------------
# PREDICTION (NON-LIVE)
# --------------------------------------------------
if data is not None and input_mode != "Live EEG (Headset)":
    with st.container():
        st.markdown(
        "<div class='glass'>",
        unsafe_allow_html=True
    )

    st.write("EEG Shape:", data.shape)

    data = z_score_normalize(data)

    tensor = torch.tensor(
        data, dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)
        pred = torch.argmax(probs).item()

    stress = float(probs[0][1])
    relax = float(probs[0][0])

    st.markdown("## 🔎 Prediction")

    if pred == 1:
        st.error("⚠️ Stress Detected")
    else:
        st.success("😊 Relaxed State")

    st.markdown("## 📊 Stress Meter")
    st.progress(stress)

    c1, c2 = st.columns(2)
    c1.metric("Relaxed", round(relax,3))
    c2.metric("Stress", round(stress,3))

    st.markdown("## 📈 EEG Signal")
    st.line_chart(data[0])

    st.markdown("</div>", unsafe_allow_html=True)
    
       
    