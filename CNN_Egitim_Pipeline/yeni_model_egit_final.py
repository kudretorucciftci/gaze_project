# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import os
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

print("DEBUG: Kütüphaneler içe aktarıldı.")

# --- Ayarlar ---
DATA_FILE = "mpiigaze_processed.npz"
MODEL_FILE = "mpiigaze_fixed.keras"
LOG_FILE = "training_log.csv"
IMAGE_SHAPE = (36, 60, 1)
BATCH_SIZE = 256
EPOCHS = 30
VALIDATION_SPLIT = 0.2

def build_model():
    """Geliştirilmiş CNN modeli oluşturur."""
    print("DEBUG: Model oluşturuluyor...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=IMAGE_SHAPE, name="conv2d_input"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(2)
    ], name="GazeTrackerModel")

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae'])
    print("DEBUG: Model derlendi.")
    return model

# TensorFlow'un resmi, verimli veri yükleme yöntemi
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_path, indices, batch_size, shuffle=True):
        print(f"DEBUG: DataGenerator başlatılıyor. Örnek sayısı: {len(indices)}")
        self.file_path = file_path
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Veri dosyasını sadece ana süreçte bir kez yükle
        _data = np.load(self.file_path, mmap_mode='r')
        self.x_data = _data['x']
        self.y_data = _data['y']
        
        self.on_epoch_end()

    def __len__(self):
        """Her epoch'taki batch sayısını döndürür."""
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, index):
        """Bir batch veri üretir."""
        # O anki batch için indis aralığını belirle
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        
        # Karıştırılmış listeden batch'e ait olacak ana indisleri al
        batch_master_indices = self.shuffled_indices[start_index:end_index]
        
        # Sadece bu batch için gerekli veriyi diskten belleğe çek
        X = self.x_data[batch_master_indices]
        y = self.y_data[batch_master_indices]
        
        return X, y

    def on_epoch_end(self):
        """Her epoch sonunda indisleri karıştırır."""
        self.shuffled_indices = self.indices.copy() # Ana indis listesini kopyala
        if self.shuffle:
            np.random.shuffle(self.shuffled_indices)

def main():
    print("--- Verimli Model Eğitimi (tf.keras.utils.Sequence ile) ---")
    
    if not os.path.exists(DATA_FILE):
        print(f"HATA: '{DATA_FILE}' bulunamadı.")
        return

    # 1. Toplam örnek sayısını verimli bir şekilde al
    with np.load(DATA_FILE, mmap_mode='r') as data:
        num_samples = data['x'].shape[0]
    print(f"Toplam {num_samples} örnek bulundu.")

    # 2. Eğitim ve Doğrulama için indisleri ayır
    all_indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(all_indices)
    
    split_point = int(VALIDATION_SPLIT * num_samples)
    val_indices = all_indices[:split_point]
    train_indices = all_indices[split_point:]

    print(f"Eğitim seti: {len(train_indices)} örnek")
    print(f"Doğrulama seti: {len(val_indices)} örnek")

    # 3. DataGenerator'ları oluştur
    print("DEBUG: Veri jeneratörleri oluşturuluyor...")
    train_generator = DataGenerator(DATA_FILE, train_indices, BATCH_SIZE, shuffle=True)
    val_generator = DataGenerator(DATA_FILE, val_indices, BATCH_SIZE, shuffle=False)

    # 4. Modeli Oluştur
    model = build_model()
    model.summary()

    # 5. Callback'leri Ayarla
    checkpoint = ModelCheckpoint(MODEL_FILE, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    csv_logger = CSVLogger(LOG_FILE)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
    callbacks_list = [checkpoint, csv_logger, early_stopping]

    # 6. Modeli Eğit
    print("\n--- Model Eğitimi Başlatılıyor ---")
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks_list
        # workers parametreleri multiprocessing sorunları nedeniyle kaldırıldı.
    )

    print("\n--- Değerlendirme ---")
    loss, mae = model.evaluate(val_generator, verbose=0)
    print(f"Doğrulama Seti Kayıp (MSE): {loss:.4f}")
    print(f"Doğrulama Seti Ortalama Mutlak Hata (MAE): {mae:.4f} (derece cinsinden yaklaşık hata)")

    print(f"\n✅ Eğitim tamamlandı! En iyi model '{MODEL_FILE}' dosyasına kaydedildi.")

if __name__ == '__main__':
    main()
