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
CSV_PATH   = "kullanici_verisi/annotations.csv"
MODEL_IN   = "../mpiigaze_finetuned.keras"
MODEL_OUT  = "../Modeller/mpiigaze_finetuned_v2.keras"

IMG_W, IMG_H = 60, 36
BATCH   = 16
EPOCHS  = 30
LR      = 1e-5

# =====================================================
# DATA LOADER (CSV ESNEK OKUMA)
# =====================================================
def load_dataset(csv_path):
    X, y = [], []

    # Get the base directory where the script is run (project root)
    project_root = os.getcwd()

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader) # Skip header row

        for row in reader:
            if len(row) < 3:
                continue

            img_relative_path = row[0].strip().replace('/', os.sep) # Normalize path separators
            img_full_path = os.path.join(project_root, img_relative_path)

            try:
                pitch = float(row[1])
                yaw   = float(row[2])
            except ValueError:
                print(f"Uyarı: Geçersiz pitch/yaw değeri satırı atlandı: {row}")
                continue

            if not os.path.exists(img_full_path):
                print(f"Uyarı: Resim dosyası bulunamadı: {img_full_path}")
                continue

            img = cv2.imread(img_full_path)
            if img is None:
                print(f"Uyarı: Resim yüklenemedi (muhtemelen bozuk): {img_full_path}")
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
    raise RuntimeError(f"❌ Yetersiz veri ({len(X)} örnek). Fine-tune için en az 50 örnek gerekli.")

# =====================================================
# MODEL LOAD
# =====================================================
print("📦 Model yükleniyor...")
model = tf.keras.models.load_model(MODEL_IN, compile=False)

# =====================================================
# FREEZE (SADECE SON KATMANLAR ÖĞRENSİN)
# =====================================================
# Temel modelin (base_model) içindeki katmanların eğitilebilirliğini ayarlayalım
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
            patience=5,
            restore_best_weights=True
        )
    ]
)

# =====================================================
# SAVE
# =====================================================
model.save(MODEL_OUT)
print(f"✅ Fine-tuned model kaydedildi → {MODEL_OUT}")
