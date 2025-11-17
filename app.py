# ============================================================================
# STREAMLIT VOICE ASSISTANT - HYBRID MODE (MIC + CLOUD SAFE)
# ============================================================================
# Works in BOTH:
#    ✔ Local computer (microphone recording via sounddevice)
#    ✔ Streamlit Cloud (switches to upload mode automatically)
#
# Microphone is used ONLY if sounddevice loads successfully.
# Otherwise, the app automatically falls back to WAV upload mode.
# ============================================================================

import streamlit as st
import os
import json
import numpy as np
import librosa
import time
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Try importing sounddevice (works locally but NOT on Streamlit Cloud)
try:
    import sounddevice as sd
    MIC_AVAILABLE = True
except Exception:
    MIC_AVAILABLE = False

# Try TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================
SAMPLE_RATE = 16000
DURATION = 1.0
N_MFCC = 40
CONFIDENCE_THRESHOLD = 0.40

# ============================================================================
# PAGE SETUP
# ============================================================================
st.set_page_config(page_title="Voice Assistant", page_icon="🎤")

st.title("🎤 Voice Assistant (Hybrid Mode)")
st.write("✔ Local: Uses microphone\n✔ Streamlit Cloud: Upload audio file")

# ============================================================================
# LOCATE MODEL FILES
# ============================================================================
def find_model_files():
    """Find voice_model.h5 and labels.json in model/ or root folder."""
    model_folder = Path.cwd() / "model"
    model_path = model_folder / "voice_model.h5"
    labels_path = model_folder / "labels.json"

    if model_path.exists() and labels_path.exists():
        return str(model_path), str(labels_path)

    # fallback
    model_path = Path.cwd() / "voice_model.h5"
    labels_path = Path.cwd() / "labels.json"
    if model_path.exists() and labels_path.exists():
        return str(model_path), str(labels_path)

    return None, None

# ============================================================================
# LOAD MODEL + LABELS
# ============================================================================
@st.cache_resource
def load_model_data():
    model_path, labels_path = find_model_files()

    if not model_path or not labels_path:
        st.error("❌ Model files not found! Place voice_model.h5 and labels.json in /model folder.")
        return None, None, None, None

    try:
        model = load_model(model_path, compile=False)
        model.compile(optimizer="adam", loss="categorical_crossentropy")

        with open(labels_path, "r") as f:
            labels = json.load(f)

        label_map = {int(k): v for k, v in labels.items()}

        # Separate intents/devices
        intent_words = {"on", "off", "start", "stop", "enable", "disable"}
        intents = [v for v in label_map.values() if v in intent_words]
        devices = [v for v in label_map.values() if v not in intent_words]

        return model, label_map, intents, devices

    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None, None, None, None

# ============================================================================
# AUDIO PROCESSING
# ============================================================================
def extract_mfcc(audio):
    """Convert raw audio to MFCC features."""
    try:
        if len(audio) < SAMPLE_RATE:
            audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)))
        else:
            audio = audio[:SAMPLE_RATE]

        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        return np.mean(mfcc.T, axis=0)
    except:
        return None

def record_microphone():
    """Record audio from microphone (LOCAL ONLY)."""
    st.info("🎤 Recording for 1 second...")
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    return audio.flatten()

# ============================================================================
# PREDICTION
# ============================================================================
def predict_label(audio, model, label_map):
    feat = extract_mfcc(audio)
    if feat is None:
        return None, 0.0

    feat = feat.reshape(1, -1)
    pred = model.predict(feat, verbose=0)[0]
    idx = int(np.argmax(pred))
    return label_map[idx], float(pred[idx])

# ============================================================================
# MAIN INTERFACE
# ============================================================================
def main():
    model, label_map, intents, devices = load_model_data()
    if model is None:
        st.stop()

    st.subheader("📋 Detected Classes")
    st.write("**Intents:**", intents)
    st.write("**Devices:**", devices)
    st.write("---")

    # MODE SELECTION
    mode = "Microphone" if MIC_AVAILABLE else "Upload File"

    if MIC_AVAILABLE:
        st.success("🎤 Microphone detected — using REAL-TIME recording")
    else:
        st.warning("⚠️ Microphone not available — using file upload mode")

    if mode == "Microphone":
        if st.button("🎙️ Record Audio (1 sec)"):
            audio = record_microphone()
            label, conf = predict_label(audio, model, label_map)

            if label and conf >= CONFIDENCE_THRESHOLD:
                st.success(f"**Recognized:** {label} ({conf:.0%})")
            else:
                st.error("❌ Could not understand audio")

    # ====================================================================
    # FILE UPLOAD MODE (for Streamlit Cloud)
    # ====================================================================
    st.write("---")
    st.subheader("📤 Upload WAV File (Cloud mode)")

    file = st.file_uploader("Upload a WAV file", type=["wav"])

    if file is not None:
        audio, _ = librosa.load(file, sr=SAMPLE_RATE)
        label, conf = predict_label(audio, model, label_map)

        if label and conf >= CONFIDENCE_THRESHOLD:
            st.success(f"**Recognized:** {label} ({conf:.0%})")
        else:
            st.error("❌ Could not classify the audio")

# RUN
if __name__ == "__main__":
    main()
