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
CSV_PATH   = "train_pitch_yaw.csv"
MODEL_IN   = "mpiigaze_fixed.keras"
MODEL_OUT  = "mpiigaze_finetuned.keras"

IMG_W, IMG_H = 60, 36
BATCH   = 16
EPOCHS  = 15
LR      = 1e-4

# =====================================================
# DATA LOADER (CSV ESNEK OKUMA)
# =====================================================
def load_dataset(csv_path):
    X, y = [], []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:

            # boş / bozuk satır
            if len(row) < 3:
                continue

            img_path = row[0].strip()
            try:
                pitch = float(row[1])
                yaw   = float(row[2])
            except:
                continue

            # conf varsa al, yoksa 1.0 kabul et
            try:
                conf = float(row[3]) if len(row) >= 4 else 1.0
            except:
                conf = 1.0

            if conf < 0.8:
                continue

            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
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
print("📥 Dataset yükleniyor...")
X, y = load_dataset(CSV_PATH)
print(f"✔ Toplam örnek: {len(X)}")

if len(X) < 50:
    raise RuntimeError("❌ Çok az veri var — fine-tune mantıklı değil")

# =====================================================
# MODEL LOAD
# =====================================================
print("📦 Model yükleniyor...")
model = tf.keras.models.load_model(MODEL_IN, compile=False)

# =====================================================
# FREEZE (SADECE SON KATMANLAR ÖĞRENSİN)
# =====================================================
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
print("🚀 Fine-tuning başlıyor...")
history = model.fit(
    X, y,
    batch_size=BATCH,
    epochs=EPOCHS,
    validation_split=0.1,
    shuffle=True,
    callbacks=[
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )
    ]
)

# =====================================================
# SAVE
# =====================================================
model.save(MODEL_OUT)
print(f"✅ Fine-tuned model kaydedildi → {MODEL_OUT}")
