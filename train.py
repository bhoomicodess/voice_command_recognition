
import os
import json
import zipfile
import urllib.request
import random
from pathlib import Path

import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000
MAX_SAMPLES = 800
EPOCHS = 20
BATCH_SIZE = 32
RANDOM_SEED = 42

LABELS = ["Start", "Stop", "Left", "Right"]

FOLDER_MAP = {
    "go": "Start",
    "stop": "Stop",
    "left": "Left",
    "right": "Right"
}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ==========================================
# 2. DOWNLOAD DATASET
# ==========================================

zip_path = DATA_DIR / "mini_speech_commands.zip"
extract_path = DATA_DIR / "mini_speech_commands"

if not extract_path.exists():

    print("Downloading Google Speech Commands dataset...")

    url = (
        "https://storage.googleapis.com/download.tensorflow.org/"
        "data/mini_speech_commands.zip"
    )

    urllib.request.urlretrieve(url, zip_path)

    print("Extracting dataset...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    print("Dataset downloaded successfully!")

# ==========================================
# 3. AUDIO LOADING AND PREPROCESSING
# ==========================================

def load_audio(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        duration=1.0,
        mono=True
    )

    target_length = SAMPLE_RATE

    if len(audio) < target_length:
        audio = np.pad(
            audio,
            (0, target_length - len(audio))
        )
    else:
        audio = audio[:target_length]

    return audio.astype(np.float32)


def augment_audio(audio):

    augmented = audio.copy()

    # Random volume adjustment
    gain = np.random.uniform(0.8, 1.2)
    augmented = augmented * gain

    # Random time shift
    max_shift = int(SAMPLE_RATE * 0.10)
    shift = np.random.randint(-max_shift, max_shift + 1)

    if shift > 0:
        augmented = np.pad(
            augmented[:-shift],
            (shift, 0)
        )

    elif shift < 0:
        shift = abs(shift)
        augmented = np.pad(
            augmented[shift:],
            (0, shift)
        )

    # Add low-level synthetic noise
    noise_level = np.random.uniform(0.001, 0.01)

    noise = np.random.normal(
        0,
        noise_level,
        len(augmented)
    )

    augmented = augmented + noise

    # Keep audio within a safe amplitude range
    augmented = np.clip(augmented, -1.0, 1.0)

    return augmented.astype(np.float32)


def extract_features(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=40,
        n_fft=512,
        hop_length=160
    )

    # Same normalization used in app.py
    mfcc = (
        mfcc - np.mean(mfcc)
    ) / (
        np.std(mfcc) + 1e-6
    )

    return mfcc.astype(np.float32)

# ==========================================
# 4. COLLECT DATASET FILE PATHS
# ==========================================

file_paths = []
labels = []

print("\nCollecting dataset recordings...")

for folder, label in FOLDER_MAP.items():

    folder_path = extract_path / folder

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder_path}"
        )

    files = list(folder_path.glob("*.wav"))

    random.shuffle(files)

    files = files[:MAX_SAMPLES]

    print(f"{label}: {len(files)} recordings")

    for file in files:
        file_paths.append(str(file))
        labels.append(LABELS.index(label))

file_paths = np.array(file_paths)
labels = np.array(labels, dtype=np.int32)

print("\nTotal original recordings:", len(file_paths))

# ==========================================
# 5. TRAIN / VALIDATION / TEST SPLIT
# ==========================================

# Split BEFORE augmentation to avoid data leakage.

X_temp, X_test_paths, y_temp, y_test = train_test_split(
    file_paths,
    labels,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=labels
)

X_train_paths, X_val_paths, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=y_temp
)

print("\nDataset split:")
print("Training recordings:", len(X_train_paths))
print("Validation recordings:", len(X_val_paths))
print("Testing recordings:", len(X_test_paths))

# ==========================================
# 6. EXTRACT TRAINING FEATURES
# ==========================================

X_train = []

print("\nExtracting training features...")

for i, file_path in enumerate(X_train_paths):

    audio = load_audio(file_path)

    original_features = extract_features(audio)

    X_train.append(original_features)

    # Augment training data only
    augmented_audio = augment_audio(audio)

    augmented_features = extract_features(
        augmented_audio
    )

    X_train.append(augmented_features)

    if (i + 1) % 100 == 0:
        print(
            f"Processed {i + 1}/{len(X_train_paths)} "
            "training recordings"
        )

# Duplicate labels for original and augmented samples
y_train_augmented = np.repeat(y_train, 2)

X_train = np.array(X_train, dtype=np.float32)

# ==========================================
# 7. EXTRACT VALIDATION FEATURES
# ==========================================

X_val = []

print("\nExtracting validation features...")

for file_path in X_val_paths:

    audio = load_audio(file_path)

    features = extract_features(audio)

    X_val.append(features)

X_val = np.array(X_val, dtype=np.float32)

# ==========================================
# 8. EXTRACT TEST FEATURES
# ==========================================

X_test = []

print("\nExtracting testing features...")

for file_path in X_test_paths:

    audio = load_audio(file_path)

    features = extract_features(audio)

    X_test.append(features)

X_test = np.array(X_test, dtype=np.float32)

# Add channel dimension for CNN
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print("\nFinal dataset shapes:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# ==========================================
# 9. BUILD IMPROVED CNN MODEL
# ==========================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=X_train.shape[1:]
    ),

    # First convolution block
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        use_bias=False
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Activation("relu"),

    tf.keras.layers.MaxPooling2D((2, 2)),

    # Second convolution block
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        use_bias=False
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Activation("relu"),

    tf.keras.layers.MaxPooling2D((2, 2)),

    # Third convolution block
    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        use_bias=False
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Activation("relu"),

    tf.keras.layers.GlobalAveragePooling2D(),

    # Classification layers
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(
        len(LABELS),
        activation="softmax"
    )
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)

model.summary()

# ==========================================
# 10. TRAINING CALLBACKS
# ==========================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(MODEL_DIR / "best_model.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

# ==========================================
# 11. TRAIN MODEL
# ==========================================

print("\nStarting improved CNN training...")

history = model.fit(
    X_train,
    y_train_augmented,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ],

    verbose=1
)

# ==========================================
# 12. EVALUATE MODEL
# ==========================================

print("\nEvaluating model on unseen test recordings...")

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

predictions = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(
    predictions,
    axis=1
)

precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

print("\n========== MODEL RESULTS ==========")

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

print("\nClassification Report:")

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3],
    target_names=LABELS,
    zero_division=0
)

print(report)

# ==========================================
# 13. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3]
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="crest",
    xticklabels=LABELS,
    yticklabels=LABELS
)

plt.xlabel("Predicted Command")
plt.ylabel("Actual Command")
plt.title("CNN Voice Command Confusion Matrix")

plt.tight_layout()

plt.savefig(
    MODEL_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()

# ==========================================
# 14. TRAINING GRAPHS
# ==========================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plt.savefig(
    MODEL_DIR / "training_graphs.png",
    dpi=300
)

plt.close()

# ==========================================
# 15. SAVE TRAINED MODEL
# ==========================================

model.save(
    MODEL_DIR / "voice_command_model.keras"
)

with open(
    MODEL_DIR / "labels.json",
    "w"
) as f:

    json.dump(LABELS, f)

# ==========================================
# 16. SAVE MODEL METRICS
# ==========================================

metrics = {
    "accuracy": float(accuracy),
    "loss": float(loss),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "training_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "testing_samples": int(len(X_test))
}

with open(
    MODEL_DIR / "metrics.json",
    "w"
) as f:

    json.dump(metrics, f, indent=4)

# ==========================================
# 17. FINAL RESULTS
# ==========================================

print("\n======================================")
print("       TRAINING COMPLETED")
print("======================================")

print("Model saved:", MODEL_DIR / "voice_command_model.keras")
print("Labels saved:", MODEL_DIR / "labels.json")
print("Metrics saved:", MODEL_DIR / "metrics.json")

print("\nFINAL PERFORMANCE")

print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1 Score:  {f1 * 100:.2f}%")

print("\nGenerated files:")
print("- voice_command_model.keras")
print("- best_model.keras")
print("- labels.json")
print("- metrics.json")
print("- confusion_matrix.png")
print("- training_graphs.png")

print("\n======================================")
