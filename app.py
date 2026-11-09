
import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import json
import io
import pandas as pd
from pathlib import Path
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="VOX AI | Voice Command Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0b1220 0%, #111c2e 55%, #101827 100%);
    color: #f1f5f9;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

.hero {
    background: linear-gradient(120deg, #172b40, #163b43);
    border: 1px solid #28535a;
    padding: 35px;
    border-radius: 24px;
    margin-bottom: 25px;
}

.hero-tag {
    color: #70e1ce;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
}

.hero-title {
    color: #ffffff;
    font-size: 42px;
    font-weight: 800;
    margin: 12px 0 8px 0;
}

.hero-subtitle {
    color: #b6c8d5;
    font-size: 16px;
}

.section-title {
    color: #f8fafc;
    font-size: 23px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 15px;
}

.metric-card {
    background: #172438;
    border: 1px solid #293d53;
    padding: 20px;
    border-radius: 17px;
    min-height: 115px;
}

.metric-label {
    color: #9fb1c5;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #70e1ce;
    font-size: 27px;
    font-weight: 800;
    margin-top: 10px;
}

.command-card {
    background: #172438;
    border: 1px solid #293d53;
    padding: 20px 12px;
    border-radius: 17px;
    text-align: center;
}

.command-icon {
    font-size: 28px;
    margin-bottom: 8px;
}

.command-name {
    color: #ffffff;
    font-weight: 800;
    font-size: 15px;
}

.command-description {
    color: #9fb1c5;
    font-size: 12px;
    margin-top: 5px;
}

.result-card {
    background: linear-gradient(135deg, #153b3d, #172c40);
    border: 1px solid #367b72;
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    margin: 15px 0;
}

.result-label {
    color: #9fded1;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
}

.result-command {
    color: #ffffff;
    font-size: 38px;
    font-weight: 800;
    margin: 8px 0;
}

.result-caption {
    color: #b9d5d1;
    font-size: 14px;
}

.info-card {
    background: #172438;
    border: 1px solid #293d53;
    border-radius: 17px;
    padding: 20px;
    color: #d6e1ec;
    min-height: 130px;
}

.footer {
    text-align: center;
    color: #8294a9;
    font-size: 12px;
    padding: 25px 0 5px 0;
}

div.stButton > button {
    background: linear-gradient(90deg, #20b9a5, #38d6bd);
    color: #071a20;
    border: none;
    border-radius: 12px;
    padding: 13px 25px;
    font-weight: 800;
    min-height: 48px;
    width: 100%;
}

div.stButton > button:hover {
    background: #65e5d1;
    color: #071a20;
    border: none;
}

div[data-testid="stAudioInput"] {
    background: #172438;
    border: 1px solid #31485e;
    border-radius: 15px;
    padding: 15px;
}

div[data-testid="stProgressBar"] > div > div {
    background: #35ceb7;
}

[data-testid="stMetric"] {
    background: #172438;
    border: 1px solid #293d53;
    padding: 15px;
    border-radius: 14px;
}

h1, h2, h3, p, label {
    color: #f1f5f9;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOAD MODEL AND PROJECT DATA
# ==========================================

@st.cache_resource
def load_model():
    model_path = MODEL_DIR / "voice_command_model.keras"
    return tf.keras.models.load_model(model_path)

@st.cache_data
def load_project_data():
    with open(MODEL_DIR / "labels.json", "r") as f:
        labels = json.load(f)

    with open(MODEL_DIR / "metrics.json", "r") as f:
        metrics = json.load(f)

    return labels, metrics

try:
    model = load_model()
    LABELS, metrics = load_project_data()
    model_ready = True
except Exception as error:
    model_ready = False
    st.error(f"Unable to load the model or project files: {error}")
    st.stop()

# ==========================================
# 4. AUDIO FEATURE EXTRACTION
# ==========================================

def extract_features(audio_bytes):

    audio, sr = librosa.load(
        io.BytesIO(audio_bytes),
        sr=16000,
        duration=1.0,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("No audio detected. Please record again.")

    target_length = 16000

    if len(audio) < target_length:
        audio = np.pad(
            audio,
            (0, target_length - len(audio))
        )
    else:
        audio = audio[:target_length]

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=16000,
        n_mfcc=40,
        n_fft=512,
        hop_length=160
    )

    mfcc = (
        mfcc - np.mean(mfcc)
    ) / (
        np.std(mfcc) + 1e-6
    )

    return mfcc[
        np.newaxis, ..., np.newaxis
    ].astype(np.float32)

# ==========================================
# 5. PREDICTION
# ==========================================

def predict_command(audio_bytes):

    features = extract_features(audio_bytes)

    probabilities = model.predict(
        features,
        verbose=0
    )[0]

    index = int(np.argmax(probabilities))

    command = LABELS[index]
    confidence = float(probabilities[index]) * 100

    return command, confidence, probabilities

# ==========================================
# 6. SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

if "result" not in st.session_state:
    st.session_state.result = None

# ==========================================
# 7. HERO SECTION
# ==========================================

st.markdown("""
<div class="hero">
    <div class="hero-tag">AI • DEEP LEARNING • SPEECH TECHNOLOGY</div>
    <div class="hero-title">VOX AI 🎙️</div>
    <div style="color:#70e1ce;font-size:19px;font-weight:600;">
        Voice Command Studio
    </div>
    <div class="hero-subtitle">
        Transform spoken words into intelligent command recognition.
        Powered by CNN and MFCC audio processing.
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 8. DASHBOARD METRICS
# ==========================================

accuracy = metrics.get("accuracy", 0) * 100

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">MODEL ACCURACY</div>
        <div class="metric-value">{accuracy:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">AI ALGORITHM</div>
        <div class="metric-value">CNN</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">AUDIO FEATURES</div>
        <div class="metric-value">MFCC</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">COMMAND CLASSES</div>
        <div class="metric-value">04</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 9. COMMAND LIBRARY
# ==========================================

st.markdown(
    '<div class="section-title">Explore Voice Commands</div>',
    unsafe_allow_html=True
)

commands = [
    ("▶️", "START", "Say GO"),
    ("⏹️", "STOP", "Say STOP"),
    ("⬅️", "LEFT", "Say LEFT"),
    ("➡️", "RIGHT", "Say RIGHT")
]

cols = st.columns(4)

for col, (icon, name, description) in zip(cols, commands):
    with col:
        st.markdown(f"""
        <div class="command-card">
            <div class="command-icon">{icon}</div>
            <div class="command-name">{name}</div>
            <div class="command-description">{description}</div>
        </div>
        """, unsafe_allow_html=True)

st.caption(
    "Important: The training dataset uses the word GO for the START command."
)

# ==========================================
# 10. VOICE RECORDING
# ==========================================

st.markdown(
    '<div class="section-title">🎤 Voice Recognition Studio</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-card">
    <b>How to use:</b><br><br>
    1. Click the microphone button below.<br>
    2. Speak one command clearly.<br>
    3. Submit your recording.<br>
    4. Click Recognize Command to see the prediction.
</div>
""", unsafe_allow_html=True)

audio_file = st.audio_input("Record your voice command")

if audio_file is not None:

    st.audio(audio_file)

    if st.button("✨  RECOGNIZE VOICE COMMAND"):

        with st.spinner("AI is analyzing your voice..."):

            try:
                command, confidence, probabilities = predict_command(
                    audio_file.getvalue()
                )

                result = {
                    "command": command,
                    "confidence": confidence,
                    "probabilities": probabilities.tolist(),
                    "time": datetime.now().strftime("%I:%M:%S %p")
                }

                st.session_state.result = result

                st.session_state.history.insert(0, {
                    "Time": result["time"],
                    "Command": command.upper(),
                    "Confidence": f"{confidence:.2f}%"
                })

                st.session_state.history = (
                    st.session_state.history[:10]
                )

            except Exception as error:
                st.error(f"Recognition failed: {error}")

# ==========================================
# 11. DISPLAY PREDICTION
# ==========================================

result = st.session_state.result

if result is not None:

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Recognition Results</div>',
        unsafe_allow_html=True
    )

    command = result["command"]
    confidence = result["confidence"]

    if confidence >= 55:

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">RECOGNIZED COMMAND</div>
            <div class="result-command">{command.upper()}</div>
            <div class="result-caption">Your voice command has been classified.</div>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.warning(
            f"Low confidence: The best prediction is {command.upper()}. "
            "Please record again in a quieter environment."
        )

    r1, r2 = st.columns(2)

    with r1:
        st.metric("Prediction Confidence", f"{confidence:.2f}%")

    with r2:
        st.metric("Predicted Command", command.upper())

    st.progress(
        min(max(confidence / 100, 0.0), 1.0)
    )

    # ======================================
    # 12. PROBABILITY CHART
    # ======================================

    st.markdown(
        '<div class="section-title">Command Probability Analysis</div>',
        unsafe_allow_html=True
    )

    probability_df = pd.DataFrame({
        "Command": LABELS,
        "Probability (%)": np.array(
            result["probabilities"]
        ) * 100
    }).set_index("Command")

    st.bar_chart(
        probability_df,
        y="Probability (%)"
    )

    st.caption(
        "The chart displays the model's predicted probability "
        "for each command. These are model scores, not guaranteed "
        "measures of correctness."
    )

# ==========================================
# 13. RECOGNITION HISTORY
# ==========================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🕘 Recent Recognition History</div>',
    unsafe_allow_html=True
)

if st.session_state.history:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    if st.button("Clear Recognition History"):
        st.session_state.history = []
        st.session_state.result = None
        st.rerun()

else:
    st.info("Your recognized commands will appear here.")

# ==========================================
# 14. ABOUT THE PROJECT
# ==========================================

st.markdown("---")

st.markdown(
    '<div class="section-title">About VOX AI</div>',
    unsafe_allow_html=True
)

a1, a2 = st.columns(2)

with a1:
    st.markdown("""
    <div class="info-card">
        <h3>🧠 Deep Learning</h3>
        The system uses a Convolutional Neural Network
        to classify voice commands into four categories.
        MFCC features represent important characteristics
        of the input audio.
    </div>
    """, unsafe_allow_html=True)

with a2:
    st.markdown("""
    <div class="info-card">
        <h3>⚙️ Technology Stack</h3>
        Python, TensorFlow, Keras, Librosa, NumPy,
        Scikit-learn and Streamlit are used to develop
        the voice recognition application.
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 15. FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    VOX AI | Voice Command Recognition System<br>
    Built using Deep Learning and Speech Processing
</div>
""", unsafe_allow_html=True)
