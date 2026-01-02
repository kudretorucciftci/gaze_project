# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

# Gereksiz uyarıları kapatarak konsolu temiz tutalım
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class GazeOptimizer:
    def __init__(self):
        # 1. Filtreleme: Mevcut akışını bozmadan sadece mikro-titremeleri temizler
        self.smooth_x = deque(maxlen=5) 
        self.smooth_y = deque(maxlen=5)
        
        # 2. Senin Çalışan Parametrelerin: Bu değerlere dokunmuyoruz
        self.yaw_center = 0.175
        self.gain_x = 75.0  # Yatay akışı sağlayan çalışan katsayın
        self.gain_y = 35.0  # Dikey akışı sağlayan çalışan katsayın
        self.x_offset = 0.02
        self.y_offset = -0.01

    def apply_smoothing(self, nx, ny):
        self.smooth_x.append(nx)
        self.smooth_y.append(ny)
        # Hareketin doğallığını bozmadan ortalama alır
        return np.mean(self.smooth_x), np.mean(self.smooth_y)

def preprocess_eye(roi):
    """Görüntü yapısını bozmadan sadece modelin beklediği formata sokar."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (60, 36))
    img = img.astype(np.float32) / 255.0
    # (36, 60) -> (1, 36, 60, 1)
    return np.expand_dims(np.expand_dims(img, axis=-1), axis=0)

def main():
    MODEL_PATH = "mpiigaze_fixed.keras"
    if not os.path.exists(MODEL_PATH):
        print(f"❌ HATA: {MODEL_PATH} bulunamadı!")
        return

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    opt = GazeOptimizer()
    
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    # Sol göz landmark seti
    L_EYE = [33, 133, 157, 158, 159, 160, 161, 246, 7, 163, 144, 145, 153, 154, 155]

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) # Ayna görüntüsü
        h, w = frame.shape[:2]
        results = mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            
            # Göz ROI (Senin çalışan kırpma mantığın)
            xs = [int(lms[i].x * w) for i in L_EYE]
            ys = [int(lms[i].y * h) for i in L_EYE]
            cx, cy = (min(xs)+max(xs))//2, (min(ys)+max(ys))//2
            ew, eh = int((max(xs)-min(xs))*2.0), int((max(ys)-min(ys))*1.6)
            roi = frame[max(cy-eh,0):min(cy+eh,h), max(cx-ew,0):min(cx+ew,w)]

            if roi.size > 0:
                input_data = preprocess_eye(roi)
                # Tahmini alırken konsola yazı yazmayı engelledik (FPS artışı)
                pred = model.predict(input_data, verbose=0)
                
                # --- ÇALIŞAN MATEMATİKSEL FORMÜLÜN (KORUNDU) ---
                # Yatayda (nx) ve dikeyde (ny) hareket sağlayan orijinal formülün:
                raw_nx = 0.5 - ((pred[0][1] - opt.yaw_center) * opt.gain_x) + opt.x_offset
                raw_ny = 0.5 + (pred[0][0] * opt.gain_y) + opt.y_offset
                
                # Geliştirme: Sadece titremeyi alan hafif yumuşatıcı
                nx, ny = opt.apply_smoothing(raw_nx, raw_ny)

                # Koordinatları ekrana dök
                tx, ty = int(np.clip(nx, 0, 1) * w), int(np.clip(ny, 0, 1) * h)
                cv2.circle(frame, (tx, ty), 12, (0, 0, 255), -1)
                cv2.circle(frame, (tx, ty), 4, (255, 255, 255), -1)

        cv2.imshow("Gaze Tracker - Stable Main", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()