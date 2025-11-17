"""
============================================================================
STREAMLIT VOICE ASSISTANT WEB APPLICATION
============================================================================
Web-based interface for the voice assistant with ON/OFF button for audio input

Run with: streamlit run app.py
============================================================================
"""

import streamlit as st
import os
import json
import time
import numpy as np
import librosa
import sounddevice as sd
from pathlib import Path
import warnings

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except:
    TENSORFLOW_AVAILABLE = False
    st.error("⚠️ TensorFlow not available. Please install: pip install tensorflow")

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================================================
# CONFIGURATION
# ============================================================================
SAMPLE_RATE = 16000
DURATION = 1.0
N_MFCC = 40
CONFIDENCE_THRESHOLD = 0.4

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Voice Assistant",
    page_icon="🎤",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
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

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
@st.cache_resource
def find_model_files():
    """Auto-detect model and labels files"""
    # Check local model folder
    local_folder = Path.cwd() / "model"
    model_path = local_folder / "voice_model.h5"
    labels_path = local_folder / "labels.json"
    
    if model_path.exists() and labels_path.exists():
        return str(model_path), str(labels_path)
    
    # Check current directory
    model_path = Path.cwd() / "voice_model.h5"
    labels_path = Path.cwd() / "labels.json"
    
    if model_path.exists() and labels_path.exists():
        return str(model_path), str(labels_path)
    
    return None, None

@st.cache_resource
def load_model_and_labels():
    """Load model and labels (cached for performance)"""
    model_path, labels_path = find_model_files()
    
    if not model_path or not labels_path:
        st.error("❌ Model files not found!")
        st.info("Please place voice_model.h5 and labels.json in the 'model/' folder")
        return None, None, None, None
    
    try:
        # Load model
        if TENSORFLOW_AVAILABLE:
            model = load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        else:
            return None, None, None, None
        
        # Load labels
        with open(labels_path, 'r') as f:
            labels_raw = json.load(f)
        
        # Parse labels
        if isinstance(labels_raw, dict):
            try:
                labels_map = {int(k): str(v) for k, v in labels_raw.items()}
            except:
                labels_map = {int(v): str(k) for k, v in labels_raw.items()}
        else:
            labels_map = {i: str(lbl) for i, lbl in enumerate(labels_raw)}
        
        # Detect intents and devices
        label_list = [labels_map[i] for i in sorted(labels_map.keys())]
        intent_tokens = {"on", "off", "start", "stop", "enable", "disable"}
        intents = [lbl for lbl in label_list if lbl.lower() in intent_tokens]
        devices = [lbl for lbl in label_list if lbl.lower() not in intent_tokens]
        
        return model, labels_map, intents, devices
    
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None, None

def extract_mfcc(audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, duration=DURATION):
    """Extract MFCC features from audio"""
    try:
        if audio is None:
            return None
        
        # Ensure correct shape
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        target_len = int(sr * duration)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=audio.astype(float), sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc.T, axis=0).astype(np.float32)
    except Exception as e:
        st.error(f"MFCC extraction error: {e}")
        return None

def record_audio(duration=DURATION, sr=SAMPLE_RATE):
    """Record audio from microphone"""
    try:
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        return audio.flatten()
    except Exception as e:
        st.error(f"Recording error: {e}")
        return None

def predict_label(audio, model, labels_map):
    """Predict label from audio"""
    feat = extract_mfcc(audio)
    if feat is None:
        return None, 0.0
    
    x = feat.reshape(1, -1)
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
# SESSION STATE INITIALIZATION
# ============================================================================
if 'listening' not in st.session_state:
    st.session_state.listening = False

if 'intent' not in st.session_state:
    st.session_state.intent = None

if 'device' not in st.session_state:
    st.session_state.device = None

if 'last_action' not in st.session_state:
    st.session_state.last_action = None

if 'command_count' not in st.session_state:
    st.session_state.command_count = 0

if 'history' not in st.session_state:
    st.session_state.history = []

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🎤 Voice Assistant</h1>
            <p>Click the button to control audio input</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, labels_map, intents, devices = load_model_and_labels()
    
    if model is None:
        st.stop()
    
    # Display detected classes
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**📋 Intents**")
        if intents:
            for intent in intents:
                st.write(f"• {intent}")
        else:
            st.write("No intents detected")
    
    with col2:
        st.info("**🔧 Devices**")
        if devices:
            for device in devices:
                st.write(f"• {device}")
        else:
            st.write("No devices detected")
    
    st.markdown("---")
    
    # Status Display
    if st.session_state.listening:
        st.markdown("""
            <div class="status-card status-listening">
                <h2>🎙️ LISTENING...</h2>
                <p>Speak your command now</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="status-card status-idle">
                <h2>🔇 IDLE</h2>
                <p>Click the button below to start</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Current State
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.intent:
            st.success(f"**Intent:** {st.session_state.intent.upper()}")
        else:
            st.warning("**Intent:** Waiting...")
    
    with col2:
        if st.session_state.device:
            st.success(f"**Device:** {st.session_state.device.upper()}")
        else:
            st.warning("**Device:** Waiting...")
    
    # Last Action
    if st.session_state.last_action:
        st.markdown(f"""
            <div class="command-result">
                {st.session_state.last_action}
            </div>
        """, unsafe_allow_html=True)
    
    # Main Control Button
    st.markdown("---")
    
    button_col1, button_col2, button_col3 = st.columns([1, 2, 1])
    
    with button_col2:
        if st.session_state.listening:
            if st.button("🔴 STOP LISTENING", type="primary", use_container_width=True):
                st.session_state.listening = False
                st.rerun()
        else:
            if st.button("🎤 START LISTENING", type="primary", use_container_width=True):
                st.session_state.listening = True
                
                # Record audio
                with st.spinner("Recording..."):
                    audio = record_audio()
                
                if audio is not None:
                    # Predict
                    with st.spinner("Processing..."):
                        label, confidence = predict_label(audio, model, labels_map)
                    
                    if label and confidence >= CONFIDENCE_THRESHOLD:
                        st.success(f"Heard: **{label}** (confidence: {confidence:.0%})")
                        
                        # Categorize
                        if intents and devices:
                            if label.lower() in [x.lower() for x in intents]:
                                st.session_state.intent = label.lower()
                                st.info(f"✅ Intent captured: **{label}**")
                            elif label.lower() in [x.lower() for x in devices]:
                                st.session_state.device = label.lower()
                                st.info(f"✅ Device captured: **{label}**")
                            
                            # Execute if both present
                            if st.session_state.intent and st.session_state.device:
                                action = f">>> {st.session_state.intent.upper()} {st.session_state.device.upper()}"
                                st.session_state.last_action = action
                                st.session_state.command_count += 1
                                
                                # Add to history
                                st.session_state.history.append({
                                    'time': time.strftime('%H:%M:%S'),
                                    'action': action
                                })
                                
                                # Reset
                                st.session_state.intent = None
                                st.session_state.device = None
                                
                                st.balloons()
                        else:
                            # Just recognize
                            st.session_state.last_action = f"Recognized: {label.upper()}"
                            st.session_state.command_count += 1
                    else:
                        if confidence < CONFIDENCE_THRESHOLD:
                            st.warning(f"⚠️ Low confidence ({confidence:.0%}). Please speak clearly.")
                        else:
                            st.error("❌ Could not process audio")
                
                st.session_state.listening = False
                time.sleep(0.3)
                st.rerun()
    
    # Reset Button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🔄 RESET STATE", use_container_width=True):
            st.session_state.intent = None
            st.session_state.device = None
            st.session_state.last_action = None
            st.rerun()
    
    # Statistics
    st.markdown("---")
    st.subheader("📊 Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Commands", st.session_state.command_count)
    
    with col2:
        st.metric("Confidence Threshold", f"{CONFIDENCE_THRESHOLD:.0%}")
    
    # History
    if st.session_state.history:
        st.markdown("---")
        st.subheader("📜 Command History")
        
        # Show last 10 commands
        for entry in reversed(st.session_state.history[-10:]):
            st.text(f"{entry['time']} - {entry['action']}")
    
    # Settings in Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Confidence threshold slider
        new_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=CONFIDENCE_THRESHOLD,
            step=0.05
        )
        
        if new_threshold != CONFIDENCE_THRESHOLD:
            globals()['CONFIDENCE_THRESHOLD'] = new_threshold
            st.success(f"Threshold updated to {new_threshold:.0%}")
        
        st.markdown("---")
        
        st.subheader("ℹ️ About")
        st.info("""
        **Voice Assistant Web App**
        
        - Click START LISTENING to record
        - Speak your command clearly
        - Wait for recognition
        - Commands execute automatically
        
        **Model Info:**
        - Sample Rate: 16kHz
        - Duration: 1 second
        - MFCC Features: 40
        """)
        
        st.markdown("---")
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.command_count = 0
            st.rerun()

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()