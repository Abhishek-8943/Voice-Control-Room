"""
============================================================================
FINAL STREAMLIT VOICE ASSISTANT APP (HYBRID VERSION)
============================================================================
✔ Works on Streamlit Cloud (upload mode)
✔ Works locally (microphone + uploads)
✔ Auto-detects model and labels
✔ 100% MFCC-shape safe (always outputs 40)
✔ No PortAudio crash
============================================================================
"""

import streamlit as st
import numpy as np
import librosa
import json
import os
from pathlib import Path
import warnings

# Try microphone recording (local only)
try:
    import sounddevice as sd
    MIC_AVAILABLE = True
except:
    MIC_AVAILABLE = False

# TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ============================================================================
# SETTINGS
# ============================================================================
SAMPLE_RATE = 16000
N_MFCC = 40
DURATION = 1.0
CONFIDENCE_THRESHOLD = 0.40

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Voice Assistant",
    page_icon="🎤",
    layout="centered"
)

st.title("🎤 Voice Assistant – Final Stable Version")
st.write("Upload audio or use microphone (local PC only)")

# ============================================================================
# FIND MODEL FILES
# ============================================================================
@st.cache_resource
def find_model():
    """Auto-detect model and labels in ./model/ or root directory."""
    paths_to_check = [
        Path("model/voice_model.h5"),
        Path("voice_model.h5"),
        Path("model/model.h5")
    ]

    label_paths = [
        Path("model/labels.json"),
        Path("labels.json")
    ]

    model_path = None
    labels_path = None

    for p in paths_to_check:
        if p.exists():
            model_path = p
            break

    for p in label_paths:
        if p.exists():
            labels_path = p
            break

    return model_path, labels_path


# ============================================================================
# LOAD MODEL + LABELS
# ============================================================================
@st.cache_resource
def load_all():
    model_path, labels_path = find_model()

    if model_path is None or labels_path is None:
        st.error("❌ Model files not found! Place voice_model.h5 and labels.json.")
        return None, None, None, None

    model = load_model(model_path, compile=False)
    model.compile(optimizer="adam", loss="categorical_crossentropy")

    with open(labels_path, "r") as f:
        raw = json.load(f)

    # Ensure correct key:int -> value:str
    labels = {int(k): v for k, v in raw.items()}

    # Intent detection
    INTENT_WORDS = {"on", "off", "start", "stop", "enable", "disable"}
    intents = [v for v in labels.values() if v.lower() in INTENT_WORDS]
    devices = [v for v in labels.values() if v.lower() not in INTENT_WORDS]

    return model, labels, intents, devices


# ============================================================================
# SAFE MFCC EXTRACTION (GUARANTEED SHAPE 40)
# ============================================================================
def extract_mfcc_safe(audio, sr=SAMPLE_RATE):
    try:
        # Stereo -> mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Force exact length = 16000
        target = int(sr * DURATION)
        if len(audio) < target:
            audio = np.pad(audio, (0, target - len(audio)))
        else:
            audio = audio[:target]

        # MFCC extraction
        mfcc = librosa.feature.mfcc(
            y=audio.astype(np.float32),
            sr=sr,
            n_mfcc=N_MFCC
        )

        # Average across time frames
        mfcc = np.mean(mfcc.T, axis=0)

        # Guarantee shape (40,)
        mfcc = np.resize(mfcc, (N_MFCC,))
        return mfcc

    except Exception as e:
        st.error(f"MFCC extraction failed: {e}")
        return None


# ============================================================================
# PREDICT FUNCTION
# ============================================================================
def predict_label(model, labels_map, audio_data):
    mfcc = extract_mfcc_safe(audio_data)
    if mfcc is None:
        return None, 0.0

    x = mfcc.reshape(1, -1)

    try:
        preds = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(preds))
        conf = float(preds[idx])
        label = labels_map.get(idx, None)
        return label, conf
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, 0.0


# ============================================================================
# RECORD AUDIO (LOCAL ONLY)
# ============================================================================
def record_audio():
    if not MIC_AVAILABLE:
        st.warning("Microphone not available on this platform.")
        return None

    try:
        audio = sd.rec(int(SAMPLE_RATE * 1), samplerate=SAMPLE_RATE,
                       channels=1, dtype="float32")
        sd.wait()
        return audio.flatten()
    except Exception as e:
        st.error(f"Microphone error: {e}")
        return None


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.subheader("📌 Model Status")
    model, labels_map, intents, devices = load_all()

    if model is None:
        st.stop()

    st.success("Model loaded successfully!")

    st.write("### Classes Detected:")
    st.write(labels_map)

    st.write("---")
    st.write("## 🎤 Choose Input Method")

    # ===== OPTION 1: MICROPHONE ============
    if MIC_AVAILABLE:
        if st.button("🎙️ Record from Microphone (Local Only)"):
            st.info("Recording...")
            audio = record_audio()
            if audio is not None:
                st.success("Audio recorded!")
                label, conf = predict_label(model, labels_map, audio)
                if label:
                    st.markdown(
                        f"### ✅ Prediction: **{label.upper()}** ({conf:.0%})")
    else:
        st.info("Microphone disabled on Streamlit Cloud.")

    st.write("---")

    # ===== OPTION 2: FILE UPLOAD ============
    uploaded = st.file_uploader("Upload .wav audio file", type=["wav"])
    if uploaded:
        st.info("Processing uploaded audio...")
        audio, _ = librosa.load(uploaded, sr=SAMPLE_RATE, mono=True)

        # Force exact length again
        if len(audio) < SAMPLE_RATE:
            audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)))
        else:
            audio = audio[:SAMPLE_RATE]

        label, conf = predict_label(model, labels_map, audio)

        if label:
            st.success(f"Prediction: **{label.upper()}** ({conf:.0%})")
        else:
            st.error("Could not classify this audio.")

    st.write("---")
    st.write("Done.")

# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    main()
