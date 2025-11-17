import streamlit as st
import os
import json
import time
import numpy as np
import librosa
from pathlib import Path
import warnings

try:
    import sounddevice as sd
    MIC_AVAILABLE = True
except Exception:
    sd = None
    MIC_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False
    st.error("⚠️ TensorFlow not available. Please install: pip install tensorflow")

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


SAMPLE_RATE = 16000
DURATION = 1.0
N_MFCC = 40
CONFIDENCE_THRESHOLD = 0.4


st.set_page_config(
    page_title="Voice Assistant",
    page_icon="🎤",
    layout="centered",
    initial_sidebar_state="collapsed"
)


st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .status-card {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .status-listening {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .status-idle {
        background-color: #f8f9fa;
        border: 2px solid #6c757d;
    }
    .command-result {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .stButton>button {
        width: 100%;
        height: 80px;
        font-size: 24px;
        font-weight: bold;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def find_model_files():
    """Auto-detect model and label files"""
    local = Path.cwd() / "model"
    mp = local / "voice_model.h5"
    lp = local / "labels.json"

    if mp.exists() and lp.exists():
        return str(mp), str(lp)

    # Check root folder
    mp = Path.cwd() / "voice_model.h5"
    lp = Path.cwd() / "labels.json"

    if mp.exists() and lp.exists():
        return str(mp), str(lp)

    return None, None


@st.cache_resource
def load_model_and_labels():
    model_path, labels_path = find_model_files()

    if not model_path or not labels_path:
        st.error("❌ Model files not found.")
        return None, None, None, None

    try:
        if not TENSORFLOW_AVAILABLE:
            return None, None, None, None

        model = load_model(model_path, compile=False)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        with open(labels_path, "r") as f:
            labels_raw = json.load(f)

        if isinstance(labels_raw, dict):
            labels_map = {int(k): str(v) for k, v in labels_raw.items()}
        else:
            labels_map = {i: str(v) for i, v in enumerate(labels_raw)}

        # Separate intents and devices
        intents = [v for v in labels_map.values() if v.lower() in ["on", "off"]]
        devices = [v for v in labels_map.values() if v.lower() not in ["on", "off"]]

        return model, labels_map, intents, devices

    except Exception as e:
        st.error(f"❌ Model loading error: {e}")
        return None, None, None, None


def extract_mfcc(audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, duration=DURATION):
    """Extract MFCC features"""
    try:
        if audio is None:
            return None

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        target = int(sr * duration)
        if len(audio) < target:
            audio = np.pad(audio, (0, target - len(audio)))
        else:
            audio = audio[:target]

        mfcc = librosa.feature.mfcc(y=audio.astype(float), sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc.T, axis=0).astype(np.float32)

    except Exception as e:
        st.error(f"MFCC error: {e}")
        return None


def record_audio(duration=DURATION, sr=SAMPLE_RATE):
    """Record audio (ONLY works locally)"""
    if not MIC_AVAILABLE:
        st.warning("🎤 Microphone not supported on Streamlit Cloud.")
        return None

    try:
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        return audio.flatten()
    except Exception as e:
        st.error(f"Microphone error: {e}")
        return None


def predict_label(audio, model, labels_map):
    feat = extract_mfcc(audio)
    if feat is None:
        return None, 0.0

    x = feat.reshape(1, -1)
    try:
        preds = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(preds))
        return labels_map[idx], float(preds[idx])
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, 0.0



def main():
    st.markdown("""
        <div class="main-header">
            <h1>🎤 Voice Assistant</h1>
            <p>Speak commands like ON / OFF for devices</p>
        </div>
    """, unsafe_allow_html=True)

    model, labels_map, intents, devices = load_model_and_labels()

    if model is None:
        st.stop()

    st.info(f"Detected classes: {list(labels_map.values())}")

    # LISTEN BUTTON
    if MIC_AVAILABLE:
        if st.button("🎤 Start Recording"):
            audio = record_audio()
            if audio is not None:
                label, conf = predict_label(audio, model, labels_map)
                st.write(f"Prediction: {label} ({conf:.2%})")
    else:
        st.warning("Microphone disabled — upload audio instead.")

    # FILE UPLOAD OPTION
    st.subheader("📁 Upload WAV File")
    file = st.file_uploader("Choose a .wav file", type=["wav"])

    if file:
        audio, sr = librosa.load(file, sr=SAMPLE_RATE)
        label, conf = predict_label(audio, model, labels_map)
        st.success(f"Prediction: {label} ({conf:.2%})")


if __name__ == "__main__":
    main()

