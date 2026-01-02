# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

# Sadece kritik hataları göster
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ==========================================
# 1. AYARLAR VE MODEL
# ==========================================
MODEL_PATH = "mpiigaze_fixed.keras"
WEBCAM_ID = 0 # Telefonun bağlı olduğu kamera ID'si

# ==========================================
# 2. GELİŞMİŞ NORMALLEŞTİRİCİ (BIAS & SMOOTHING)
# ==========================================
class AdvancedGazeManager:
    def __init__(self):
        self.history_x = deque(maxlen=10) # Yumuşatma için son 10 kare
        self.history_y = deque(maxlen=10)
        
        # --- [AYAR BÖLGESİ 1]: MERKEZLEME VE KAZANÇ ---
        self.yaw_center = 0.175  # Önceki verilerine göre orta nokta
        self.gain_x = 75.0       # Yatay hassasiyet (Telefon için yüksek)
        self.gain_y = 25.0       # Dikey hassasiyet
        
        # --- [AYAR BÖLGESİ 2]: BIAS CORRECTION (SAPMA DÜZELTME) ---
        # Nokta hep SOLDA kalıyorsa artır (+), SAĞDA ise azalt (-)
        self.x_offset = 0.02 
        # Nokta hep YUKARIDA kalıyorsa artır (+), AŞAĞIDA ise azalt (-)
        self.y_offset = -0.01 

    def update(self, p_raw, y_raw):
        # Ham hesaplama
        nx = 0.5 - ((y_raw - self.yaw_center) * self.gain_x) + self.x_offset
        ny = 0.5 + (p_raw * self.gain_y) + self.y_offset
        
        # Geçmişe ekle ve ortalama al (Titremeyi engeller)
        self.history_x.append(nx)
        self.history_y.append(ny)
        
        return np.clip(np.mean(self.history_x), 0, 1), np.clip(np.mean(self.history_y), 0, 1)

# ==========================================
# 3. YARDIMCI FONKSİYONLAR
# ==========================================
def preprocess_eye(roi):
    """Göz bölgesini model için iyileştirir (CLAHE Uygular)"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(gray)
    img = cv2.resize(img, (60, 36))
    img = img.astype(np.float32) / 255.0
    
    # SADECE BURAYI DÜZELTTİK:
    # (36, 60) -> (36, 60, 1) -> (1, 36, 60, 1)
    img = np.expand_dims(img, axis=-1)   # Kanal ekle (Son boyut)
    img = np.expand_dims(img, axis=0)    # Batch ekle (İlk boyut)
    return img

# ==========================================
# 4. ANA DÖNGÜ
# ==========================================
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ HATA: {MODEL_PATH} bulunamadı!")
        return

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    manager = AdvancedGazeManager()
    
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Göz İndeksleri
    L_EYE = [33, 133, 157, 158, 159, 160, 161, 246, 7, 163, 144, 145, 153, 154, 155]
    R_EYE = [362, 263, 384, 385, 386, 387, 388, 466, 249, 390, 373, 374, 380, 381, 382]

    cap = cv2.VideoCapture(WEBCAM_ID)
    print("🚀 SİSTEM BAŞLADI. ÇIKIŞ İÇİN 'ESC' BASIN.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1) # Aynalama
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb)

        nx, ny = 0.5, 0.5

        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            
            p_preds, y_preds = [], []
            
            # İki gözü de işle (Çift Göz Desteği)
            for eye_indices in [L_EYE, R_EYE]:
                xs = [int(lms[i].x * w) for i in eye_indices]
                ys = [int(lms[i].y * h) for i in eye_indices]
                x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                
                # Dinamik ROI (Geniş açılı telefon için optimize)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                ew, eh = int((x2-x1)*2.0), int((y2-y1)*1.6)
                roi = frame[max(cy-eh,0):min(cy+eh,h), max(cx-ew,0):min(cx+ew,w)]
# main içindeki ilgili satır şuna benzemeli:
                if roi.size > 0:
                    input_img = preprocess_eye(roi) # Bu fonksiyon artık (1, 36, 60, 1) döndürüyor
                    prediction = model.predict(input_img, verbose=0)
                    p_preds.append(prediction[0][0])
                    y_preds.append(prediction[0][1])

            if p_preds:
                # İki gözün ortalamasını alarak hatayı düşür
                avg_p = np.mean(p_preds)
                avg_y = np.mean(y_preds)
                nx, ny = manager.update(avg_p, avg_y)

        # Görselleştirme
        target_x, target_y = int(nx * w), int(ny * h)
        cv2.circle(frame, (target_x, target_y), 15, (0, 0, 255), -1) # Dış kırmızı
        cv2.circle(frame, (target_x, target_y), 6, (255, 255, 255), -1) # İç beyaz
        
        cv2.putText(frame, f"Bias X:{manager.x_offset} Y:{manager.y_offset}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("GAZE PRO - ADVANCED", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()