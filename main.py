"""
============================================================================
VOICE ASSISTANT TRAINING NOTEBOOK - GOOGLE COLAB
============================================================================
Complete training pipeline for speech command recognition
Run this entire notebook in Google Colab from top to bottom

Instructions:
1. Copy this entire code into a new Google Colab notebook
2. Run all cells sequentially (Runtime → Run all)
3. Download the generated files when prompted
4. Place files in your local project's model/ folder
============================================================================
"""

# ============================================================================
# CELL 1: Install Required Packages
# ============================================================================
print("=" * 70)
print("STEP 1: Installing Required Packages")
print("=" * 70)
!pip install -q librosa tensorflow scikit-learn
print("✅ All packages installed successfully!\n")

# ============================================================================
# CELL 2: Mount Google Drive
# ============================================================================
print("=" * 70)
print("STEP 2: Mounting Google Drive")
print("=" * 70)
from google.colab import drive
drive.mount('/content/drive')
print("✅ Google Drive mounted successfully!\n")

# ============================================================================
# CELL 3: Import All Required Libraries
# ============================================================================
print("=" * 70)
print("STEP 3: Importing Libraries")
print("=" * 70)
import os
import json
import numpy as np
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully!\n")

# ============================================================================
# CELL 4: Configure Dataset Path
# ============================================================================
print("=" * 70)
print("STEP 4: Configuring Dataset Path")
print("=" * 70)

# Set your dataset base path here
BASE_PATH = "/content/drive/MyDrive/Speech Commands Dataset"

print(f"Dataset base path: {BASE_PATH}")
print(f"Checking if path exists... ", end="")

if os.path.exists(BASE_PATH):
    print("✅ Found!")
else:
    print("❌ NOT FOUND!")
    print("\n⚠️  Please update BASE_PATH to match your Google Drive structure")
    print("Example: '/content/drive/MyDrive/YourFolder/Speech Commands Dataset'")

print()

# ============================================================================
# CELL 5: Auto-Detect Dataset Folders
# ============================================================================
print("=" * 70)
print("STEP 5: Searching for .wav Files")
print("=" * 70)

def find_wav_folders(root_path):
    """Recursively find all folders containing .wav files"""
    wav_folders = []

    if not os.path.exists(root_path):
        print(f"❌ Base path not found: {root_path}")
        return wav_folders

    print(f"Scanning: {root_path}")

    for dirpath, dirnames, filenames in os.walk(root_path):
        # Check if this folder contains .wav files
        wav_files = [f for f in filenames if f.lower().endswith('.wav')]
        if wav_files:
            wav_folders.append((dirpath, len(wav_files)))

    return wav_folders

# Find all folders with wav files
wav_folders = find_wav_folders(BASE_PATH)

if not wav_folders:
    print(f"\n❌ ERROR: No .wav files found in {BASE_PATH}")
    print("Please check:")
    print("  1. Your Google Drive path is correct")
    print("  2. Audio files are .wav format")
    print("  3. Files are in the expected location")
else:
    print(f"\n✅ Found {len(wav_folders)} folder(s) containing .wav files:\n")
    for folder, count in wav_folders:
        print(f"  📁 {folder}")
        print(f"     └─ {count} files\n")

# Select the folder to use (modify if you have multiple)
if wav_folders:
    DATASET_PATH = wav_folders[0][0]
    print(f"📌 Using dataset from: {DATASET_PATH}\n")
else:
    raise ValueError("Cannot proceed without dataset!")

# ============================================================================
# CELL 6: Analyze Dataset Structure
# ============================================================================
print("=" * 70)
print("STEP 6: Analyzing Dataset Structure")
print("=" * 70)

def get_class_distribution(dataset_path):
    """Get class labels and file counts"""
    class_counts = {}

    # Check if files are organized in class subfolders
    try:
        items = os.listdir(dataset_path)
        subdirs = [d for d in items if os.path.isdir(os.path.join(dataset_path, d))]
    except Exception as e:
        print(f"Error accessing dataset: {e}")
        return class_counts

    if subdirs:
        # Files organized in class folders
        print("Dataset structure: CLASS FOLDERS (organized by subfolder)")
        for class_name in subdirs:
            class_path = os.path.join(dataset_path, class_name)
            try:
                wav_files = [f for f in os.listdir(class_path)
                           if f.lower().endswith('.wav')]
                if wav_files:
                    class_counts[class_name.lower()] = len(wav_files)
            except:
                continue
    else:
        # Files in root folder - extract class from filename
        print("Dataset structure: FLAT (files named with class prefix)")
        wav_files = [f for f in items if f.lower().endswith('.wav')]
        for fname in wav_files:
            # Assuming format like "on_001.wav" or "fan_002.wav"
            parts = fname.split('_')
            if parts:
                class_name = parts[0].lower()
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

    return class_counts

class_counts = get_class_distribution(DATASET_PATH)

print(f"\n📊 Classes found: {len(class_counts)}\n")
total_samples = 0
classes_to_remove = []

# First pass - identify classes to remove
for cls, count in sorted(class_counts.items()):
    status = ""
    if count < 2:
        status = " ⚠️  TOO FEW - WILL BE REMOVED"
        classes_to_remove.append(cls)
    elif count < 10:
        status = " ⚠️  Limited samples"
    print(f"  • {cls:10s} : {count:4d} samples{status}")
    if count >= 2:
        total_samples += count

# Remove invalid classes
if classes_to_remove:
    print(f"\n⚠️  Removing {len(classes_to_remove)} classes with < 2 samples:")
    print(f"   {classes_to_remove}")
    for cls in classes_to_remove:
        del class_counts[cls]
    print(f"\n✅ Remaining valid classes: {sorted(class_counts.keys())}")
    print(f"   Total valid samples: {total_samples}")
else:
    print(f"\n✅ All classes have sufficient samples")
    print(f"   Total samples: {total_samples}")

# Check if we have enough valid classes
if len(class_counts) < 2:
    raise ValueError(
        f"Not enough valid classes! Found only {len(class_counts)} class(es) with ≥2 samples. "
        "Need at least 2 classes for classification. Please add more training data."
    )

print()

if total_samples < 20:
    print("⚠️  Warning: Very few samples detected. Model may not train well.")
elif total_samples < 100:
    print("⚠️  Warning: Limited samples. Consider recording more for better accuracy.")
else:
    print("✅ Good amount of training data!")

print()

# ============================================================================
# CELL 7: Define Feature Extraction Function
# ============================================================================
print("=" * 70)
print("STEP 7: Setting Up Feature Extraction")
print("=" * 70)

# Audio processing parameters
SAMPLE_RATE = 16000
DURATION = 1.0
N_MFCC = 40

print(f"Audio Configuration:")
print(f"  • Sample Rate: {SAMPLE_RATE} Hz")
print(f"  • Duration: {DURATION} seconds")
print(f"  • MFCC Coefficients: {N_MFCC}")
print()

def extract_mfcc(file_path, sr=SAMPLE_RATE, duration=DURATION, n_mfcc=N_MFCC):
    """Extract MFCC features from audio file"""
    try:
        # Load audio file
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)

        # Pad if too short
        target_length = int(sr * duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))

        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        return mfcc_mean
    except Exception as e:
        return None

print("✅ Feature extraction function ready!\n")

# ============================================================================
# CELL 8: Load and Process Dataset
# ============================================================================
print("=" * 70)
print("STEP 8: Loading and Processing Audio Files")
print("=" * 70)
print("This may take a few minutes depending on dataset size...\n")

def load_dataset(dataset_path, valid_classes=None):
    """Load all audio files and extract features"""
    features = []
    labels = []
    failed_files = []

    # Check dataset structure
    try:
        items = os.listdir(dataset_path)
        subdirs = [d for d in items if os.path.isdir(os.path.join(dataset_path, d))]
    except Exception as e:
        print(f"Error accessing dataset: {e}")
        return np.array([]), np.array([])

    if subdirs:
        # Class-organized structure
        print("Processing class-organized dataset...\n")
        for class_name in sorted(subdirs):
            class_name_lower = class_name.lower()

            # Skip classes with too few samples
            if valid_classes and class_name_lower not in valid_classes:
                print(f"⏭️  Skipping '{class_name}' (too few samples)")
                continue

            class_path = os.path.join(dataset_path, class_name)
            try:
                wav_files = [f for f in os.listdir(class_path)
                           if f.lower().endswith('.wav')]
            except:
                continue

            print(f"📂 Processing '{class_name}': {len(wav_files)} files... ", end="")

            processed = 0
            for wav_file in wav_files:
                file_path = os.path.join(class_path, wav_file)
                mfcc_features = extract_mfcc(file_path)

                if mfcc_features is not None:
                    features.append(mfcc_features)
                    labels.append(class_name_lower)
                    processed += 1
                else:
                    failed_files.append(file_path)

            print(f"✓ ({processed} successful)")
    else:
        # Files in root - extract class from filename
        print("Processing flat dataset...\n")
        wav_files = [f for f in items if f.lower().endswith('.wav')]

        for i, wav_file in enumerate(wav_files):
            if i % 50 == 0:
                print(f"  Processed {i}/{len(wav_files)} files...")

            file_path = os.path.join(dataset_path, wav_file)
            class_name = wav_file.split('_')[0].lower()

            # Skip classes with too few samples
            if valid_classes and class_name not in valid_classes:
                continue

            mfcc_features = extract_mfcc(file_path)

            if mfcc_features is not None:
                features.append(mfcc_features)
                labels.append(class_name)
            else:
                failed_files.append(file_path)

        print(f"  Processed {len(wav_files)}/{len(wav_files)} files... ✓")

    if failed_files:
        print(f"\n⚠️  {len(failed_files)} files failed to process")

    return np.array(features), np.array(labels)

# Load the dataset (only valid classes)
valid_classes = set(class_counts.keys())
X, y = load_dataset(DATASET_PATH, valid_classes)

print(f"\n{'=' * 70}")
print("Feature Extraction Complete!")
print("=" * 70)
print(f"  • Total samples processed: {len(X)}")
print(f"  • Feature vector shape: {X.shape}")
print(f"  • Unique classes: {len(np.unique(y))}")
print(f"  • Class labels: {sorted(np.unique(y))}")
print()

if len(X) == 0:
    raise ValueError("No features extracted! Check your audio files.")

# Verify class distribution and remove any classes with < 2 samples
print("Verifying final class distribution:")
unique, counts = np.unique(y, return_counts=True)

# Find classes that still have < 2 samples
invalid_classes = []
for cls, count in zip(unique, counts):
    print(f"  • {cls}: {count} samples", end="")
    if count < 2:
        print(" ⚠️  INVALID - Removing")
        invalid_classes.append(cls)
    else:
        print()

# Remove samples from invalid classes
if invalid_classes:
    print(f"\n⚠️  Removing {len(invalid_classes)} invalid class(es): {invalid_classes}")
    valid_mask = ~np.isin(y, invalid_classes)
    X = X[valid_mask]
    y = y[valid_mask]
    print(f"✅ Removed invalid samples. Remaining: {len(X)} samples")

    # Re-verify distribution
    print("\nFinal verified distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  • {cls}: {count} samples")

# Final validation
if len(np.unique(y)) < 2:
    raise ValueError(
        f"Not enough valid classes after filtering! Only {len(np.unique(y))} class(es) remain. "
        "Need at least 2 classes. Please add more diverse training data."
    )

print()

# ============================================================================
# CELL 9: Prepare Data for Training
# ============================================================================
print("=" * 70)
print("STEP 9: Preparing Training Data")
print("=" * 70)

# Encode labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Create label mapping
label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}

print("Label Encoding Mapping:")
for idx, label in label_mapping.items():
    print(f"  {idx} → '{label}'")
print()

# Check if we have enough samples per class for splitting
min_samples_per_class = min(np.bincount(y_encoded))
print(f"Minimum samples per class: {min_samples_per_class}")

# This should never happen now, but keep as safety check
if min_samples_per_class < 2:
    print("\n❌ CRITICAL ERROR: Found class with < 2 samples after filtering!")
    print("This shouldn't happen. Checking class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  • {cls}: {count} samples")
    raise ValueError(
        f"Data consistency error. Please check your dataset and re-run from the beginning."
    )

# Determine test size based on dataset size
if len(X) < 50:
    test_size = 0.15  # Use smaller test set for small datasets
    print(f"⚠️  Small dataset detected, using test_size={test_size}")
else:
    test_size = 0.2

# Split dataset into train and test
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical,
        test_size=test_size,
        random_state=42,
        stratify=y_encoded
    )
    print(f"✅ Successfully split dataset")
except ValueError as e:
    print(f"⚠️  Stratified split failed: {e}")
    print("   Trying non-stratified split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical,
        test_size=test_size,
        random_state=42
    )
    print(f"✅ Non-stratified split successful")

print(f"\nDataset Split:")
print(f"  • Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"  • Testing samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
print()

# ============================================================================
# CELL 10: Build Neural Network Model
# ============================================================================
print("=" * 70)
print("STEP 10: Building Neural Network")
print("=" * 70)

num_classes = len(label_encoder.classes_)
input_shape = X_train.shape[1]

print(f"Model Configuration:")
print(f"  • Input features: {input_shape}")
print(f"  • Output classes: {num_classes}")
print()

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(input_shape,)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    Dropout(0.2),

    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model Architecture:")
print("-" * 70)
model.summary()
print("-" * 70)
print()

# ============================================================================
# CELL 11: Train the Model
# ============================================================================
print("=" * 70)
print("STEP 11: Training Model")
print("=" * 70)
print("Training in progress... This may take several minutes.\n")

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# Adjust epochs and batch size for small datasets
if len(X_train) < 100:
    epochs = 150
    batch_size = min(16, len(X_train) // 5)
    print(f"⚠️  Small dataset: Using epochs={epochs}, batch_size={batch_size}\n")
else:
    epochs = 100
    batch_size = 32

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping],
    verbose=1
)

print("\n" + "=" * 70)
print("Training Complete!")
print("=" * 70)

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\n📊 Final Model Performance:")
print(f"  • Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  • Test Loss: {test_loss:.4f}")

if test_accuracy > 0.9:
    print("  • Status: ✅ Excellent!")
elif test_accuracy > 0.75:
    print("  • Status: ✅ Good")
elif test_accuracy > 0.6:
    print("  • Status: ⚠️  Fair - Consider more training data")
else:
    print("  • Status: ⚠️  Poor - Check dataset quality")

print()

# ============================================================================
# CELL 12: Save Model and Labels
# ============================================================================
print("=" * 70)
print("STEP 12: Saving Model Files")
print("=" * 70)

# Save model
MODEL_FILE = 'voice_model.h5'
model.save(MODEL_FILE)
print(f"✅ Saved model: {MODEL_FILE}")

# Save label mapping
LABELS_FILE = 'labels.json'
with open(LABELS_FILE, 'w') as f:
    json.dump(label_mapping, f, indent=2)
print(f"✅ Saved labels: {LABELS_FILE}")

# Verify files exist
if os.path.exists(MODEL_FILE) and os.path.exists(LABELS_FILE):
    print(f"\n✅ Both files ready for download!")
    print(f"  • {MODEL_FILE}: {os.path.getsize(MODEL_FILE)/1024:.2f} KB")
    print(f"  • {LABELS_FILE}: {os.path.getsize(LABELS_FILE)} bytes")
else:
    print("\n❌ Error: Files not saved properly")

print()

# ============================================================================
# CELL 13: Download Files
# ============================================================================
print("=" * 70)
print("STEP 13: Downloading Files to Your Computer")
print("=" * 70)

from google.colab import files

print("Downloading files... (check your browser's download folder)\n")

try:
    # Download model
    print(f"📥 Downloading {MODEL_FILE}... ", end="")
    files.download(MODEL_FILE)
    print("✓")

    # Download labels
    print(f"📥 Downloading {LABELS_FILE}... ", end="")
    files.download(LABELS_FILE)
    print("✓")

    print("\n✅ Downloads complete!")
except Exception as e:
    print(f"\n❌ Download error: {e}")
    print("You can manually download the files from the Colab file browser")

print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("🎉 TRAINING COMPLETE!")
print("=" * 70)
print("\n📋 Summary:")
print(f"  • Model trained on {len(X_train)} samples")
print(f"  • Test accuracy: {test_accuracy*100:.2f}%")
print(f"  • Classes: {sorted(label_mapping.values())}")
print(f"  • Files generated: {MODEL_FILE}, {LABELS_FILE}")
print("\n📦 Next Steps:")
print("  1. Check your Downloads folder for the two files")
print("  2. Create a 'model/' folder in your VS Code project")
print("  3. Move both files into the model/ folder")
print("  4. Run realtime_assistant.py in VS Code")
print("  5. Start speaking commands!")
print("\n" + "=" * 70)
print("Happy voice commanding! 🎙️")
print("=" * 70)