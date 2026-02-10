# -*- coding: utf-8 -*-
import os
import cv2
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# =====================================================
# CONFIG
# =====================================================
CSV_PATH   = "user_data/annotations.csv"
MODEL_IN   = "../mpiigaze_finetuned_v2.keras"
MODEL_OUT  = "../models/mpiigaze_finetuned_v2.keras"

IMG_W, IMG_H = 60, 36
BATCH   = 16
EPOCHS  = 30
LR      = 1e-5

# =====================================================
# DATA LOADER (CSV FLEXIBLE READING)
# =====================================================
def load_dataset(csv_path):
    X, y = [], []

    # Get the base directory where the script is run (project root)
    project_root = os.getcwd()

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            next(reader) # Skip header row
        except StopIteration:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        for row in reader:
            if len(row) < 3:
                continue

            img_relative_path = row[0].strip().replace('/', os.sep) # Normalize path separators
            img_full_path = os.path.join(project_root, img_relative_path)

            try:
                pitch = float(row[1])
                yaw   = float(row[2])
            except ValueError:
                print(f"Warning: Invalid pitch/yaw value line skipped: {row}")
                continue

            if not os.path.exists(img_full_path):
                print(f"Warning: Image file not found: {img_full_path}")
                continue

            img = cv2.imread(img_full_path)
            if img is None:
                print(f"Warning: Image could not be loaded (possibly corrupted): {img_full_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (IMG_W, IMG_H))
            gray = gray.astype(np.float32) / 255.0

            X.append(gray[..., None])
            y.append([pitch, yaw])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# =====================================================
# MAIN
# =====================================================
print("📥 Loading dataset...")
X, y = load_dataset(CSV_PATH)
print(f"✔ Total samples: {len(X)}")

if len(X) < 50:
    print(f"❌ Insufficient data ({len(X)} samples). At least 50 samples are required for fine-tuning.")
    exit(1)

# =====================================================
# MODEL LOAD
# =====================================================
print("📦 Loading model...")
model = tf.keras.models.load_model(MODEL_IN, compile=False)

# =====================================================
# FREEZE (ONLY LAST LAYERS LEARN)
# =====================================================
# Set adjustability of layers within the base model
for layer in model.layers:
    name = layer.name.lower()
    if "conv" in name or "bn" in name:
        layer.trainable = False
    else:
        layer.trainable = True

# =====================================================
# COMPILE
# =====================================================
model.compile(
    optimizer=Adam(learning_rate=LR),
    loss="mse"
)

model.summary()

# =====================================================
# TRAIN
# =====================================================
print("🚀 Fine-tuning starting...")
history = model.fit(
    X, y,
    batch_size=BATCH,
    epochs=EPOCHS,
    validation_split=0.1,
    shuffle=True,
    callbacks=[
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ]
)

# =====================================================
# SAVE
# =====================================================
if not os.path.exists(os.path.dirname(MODEL_OUT)):
    os.makedirs(os.path.dirname(MODEL_OUT))

model.save(MODEL_OUT)
print(f"✅ Fine-tuned model saved → {MODEL_OUT}")
