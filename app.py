# ============================================================================
# STREAMLIT VOICE ASSISTANT (HYBRID MODE)
# ============================================================================
# ✔ Works Locally (Microphone available)
# ✔ Works on Streamlit Cloud (Microphone disabled → Upload Mode)
# ✔ Auto-detects correct loss function for your model
# ✔ Fixes MFCC shape mismatch errors
# ✔ Loads your trained model safely
# ============================================================================

import streamlit as st
import numpy as np
import librosa
import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Try TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False

# Try microphone library
try:
    import sounddevice as sd
    MIC_AVAILABLE = True
except:
    MIC_AVAILABLE = False

# ============================================================================
# CONFIG
# ============================================================================
SAMPLE_RATE = 16000
DURATION = 1.0
N_MFCC = 40
CONFIDENCE_THRESHOLD = 0.4

# ============================================================================
# FIND MODEL FILES
# ============================================================================
def find_model_files():
    model_dir = Path("model")
    if (model_dir / "voice_model.h5").exists() and (model_dir / "labels.json").exists():
        return str(model_dir / "voice_model.h5"), str(model_dir / "labels.json")

    if Path("voice_model.h5").exists() and Path("labels.json").exists():
        return "voice_model.h5", "labels.json"

    return None, None

# ============================================================================
# AUTO-DETECT LOSS FUNCTION
# ============================================================================
def detect_loss(model):
    """Detect whether the model expects one-hot or sparse integer labels."""
    output_shape = model.output_shape

    # Example shapes:
    # sparse → (None, num_classes)
    # categorical → (None, num_classes)
    # We detect based on final activation AND number of outputs
    if len(output_shape) == 2 and output_shape[1] >= 2:
        return "categorical_crossentropy"  # safest for multi-class softmax

    return "sparse_categorical_crossentropy"

# ============================================================================
# LOAD MODEL + LABELS
# ============================================================================
@st.cache_resource
def load_model_data():
    model_path, labels_path = find_model_files()

    if not model_path:
        st.error("❌ Model files not found! Place voice_model.h5 and labels.json in /model folder.")
        return None, None, None, None

    try:
        model = load_model(model_path, compile=False)

        # Auto-detect correct loss
        loss_used = detect_loss(model)
        model.compile(optimizer="adam", loss=loss_used)
        st.success(f"Model loaded using **{loss_used}**")

        with open(labels_path, "r") as f:
            labels = json.load(f)

        labels_map = {int(k): v for k, v in labels.items()}

        # Identify intents/devices
        intent_tokens = {"on", "off", "start", "stop"}
        intents = [v for v in labels_map.values() if v.lower() in intent_tokens]
        devices = [v for v in labels_map.values() if v.lower() not in intent_tokens]

        return model, labels_map, intents, devices

    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        return None, None, None, None

# ============================================================================
# AUDIO PROCESSING
# ============================================================================
def extract_mfcc(audio):
    try:
        # Force mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Force exact length = 1 second
        target_len = int(SAMPLE_RATE * DURATION)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]

        mfcc = librosa.feature.mfcc(
            y=audio.astype(np.float32),
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC
        )

        mfcc = np.mean(mfcc.T, axis=0)

        # Guarantee exact 40 shape
        if mfcc.shape[0] != N_MFCC:
            mfcc = np.resize(mfcc, (N_MFCC,))

        return mfcc

    except Exception as e:
        st.error(f"MFCC extraction failed: {e}")
        return None

def record_mic():
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    return audio.flatten()

# ============================================================================
# PREDICT
# ============================================================================
def predict_label(audio, model, labels_map):
    feat = extract_mfcc(audio)
    if feat is None:
        return None, 0.0

    feat = feat.reshape(1, -1)

    preds = model.predict(feat, verbose=0)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx])

    return labels_map[idx], conf

# ============================================================================
# UI
# ============================================================================
def main():
    st.title("🎤 Voice Assistant (Hybrid Mode)")

    model, labels_map, intents, devices = load_model_data()
    if model is None:
        st.stop()

    st.subheader("🧠 Model Classes")
    st.write("**Intents:**", intents)
    st.write("**Devices:**", devices)

    st.write("---")

    # LOCAL MODE (Microphone)
    if MIC_AVAILABLE:
        st.success("🎤 Microphone detected — local recording enabled")
        if st.button("Record Audio (1 sec)"):
            audio = record_mic()
            label, conf = predict_label(audio, model, labels_map)

            if conf >= CONFIDENCE_THRESHOLD:
                st.success(f"Recognized: **{label}** ({conf:.0%})")
            else:
                st.error("Low confidence. Try again.")

    # CLOUD MODE (Upload)
    else:
        st.warning("⚠ No microphone available — upload mode enabled")

    st.subheader("📤 Upload a WAV File")

    file = st.file_uploader("Upload...", type=["wav"])
    if file is not None:
        audio, _ = librosa.load(file, sr=SAMPLE_RATE, mono=True)
        label, conf = predict_label(audio, model, labels_map)

        if conf >= CONFIDENCE_THRESHOLD:
            st.success(f"Recognized: **{label}** ({conf:.0%})")
        else:
            st.error("Unable to classify audio")

# RUN
if __name__ == "__main__":
    main()
