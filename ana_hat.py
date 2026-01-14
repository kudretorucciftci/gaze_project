# -*- coding: utf-8 -*-
import os
import cv2
import sys
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyautogui
import time
from collections import deque

from gaze_a3_a4_a5 import TemporalSmoother, BiasMap, IntentDetector
from fixation_logger import FixationLogger

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
ONLINE_LEARNING = True

# =====================================================
class GazeFlowManager:
    def __init__(self):
        self.yaw_center = None
        self.nx = 0.5
        self.ny = 0.5
        self.lin_gain_x = 2.0
        self.lin_gain_y = 1.5
        self.non_gain_x = 28.0
        self.non_gain_y = 18.0
        self.interp = 0.22

    def update(self, pitch, yaw):
        if self.yaw_center is None:
            self.yaw_center = yaw

        dx = yaw - self.yaw_center
        dy = pitch
        r = np.sqrt(dx*dx + dy*dy)

        tx_lin = 0.5 - dx * self.lin_gain_x
        ty_lin = 0.5 + dy * self.lin_gain_y

        mx = np.sign(dx) * (abs(dx) ** 0.75) * self.non_gain_x
        my = np.sign(dy) * (abs(dy) ** 0.75) * self.non_gain_y

        tx_non = 0.5 - mx
        ty_non = 0.5 + my

        w = np.clip((r - 0.04) / 0.12, 0, 1)
        tx = (1 - w) * tx_lin + w * tx_non
        ty = (1 - w) * ty_lin + w * ty_non

        self.nx += (tx - self.nx) * self.interp
        self.ny += (ty - self.ny) * self.interp

        return np.clip(self.nx, 0, 1), np.clip(self.ny, 0, 1)

# =====================================================
class PupilTracker:
    def __init__(self):
        self.last = None
        self.smooth = 0.45

    def get(self, iris, w, h):
        pts = np.array([[p.x * w, p.y * h] for p in iris])
        c = pts.mean(axis=0)

        if self.last is None:
            self.last = c
            return 0, 0, 0

        delta = c - self.last
        self.last = self.last * self.smooth + c * (1 - self.smooth)
        return delta[0] / w, delta[1] / h, np.linalg.norm(delta)

# =====================================================
class GazeIntentFilter:
    def __init__(self):
        self.hist = deque(maxlen=10)
        self.intent = 0.0

    def confidence(self, nx, ny, mag):
        self.hist.append((nx, ny))

        if len(self.hist) < 5:
            self.intent *= 0.8
            return self.intent

        xs = [p[0] for p in self.hist]
        ys = [p[1] for p in self.hist]
        std = np.sqrt(np.var(xs) + np.var(ys))

        spatial = np.clip(1 - std * 130, 0, 1)
        pupil = np.exp(-((mag - 0.003) ** 2) / 0.000015)

        raw = 0.6 * spatial + 0.4 * pupil

        if raw > self.intent:
            self.intent += (raw - self.intent) * 0.35
        else:
            self.intent *= 0.75

        return np.clip(self.intent, 0, 1)

# =====================================================
def preprocess_eye(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (60, 36)).astype(np.float32) / 255.0
    return img[None, ..., None]

# =====================================================
def main():
    # --- Kurulum ---
    pyautogui.FAILSAFE = False
    screen_w, screen_h = pyautogui.size()
    print(f"Ekran Çözünürlüğü: {screen_w}x{screen_h}")

    # --- Akıcı Kaydırma Ayarları ---
    SCROLL_ZONE_HEIGHT = 70  # Ekranın üst/altındaki aktif bölge yüksekliği (piksel)
    SCROLL_ACTIVATION_DWELL = 0.6  # Kaydırmayı başlatmak için bekleme süresi (saniye)
    SCROLL_SPEED = 40  # Kaydırma hızı (pozitif = aşağı, negatif = yukarı)
    SCROLL_COOLDOWN = 0.05  # Kaydırma komutları arası bekleme süresi
    last_scroll_time = 0

    # --- Göz Hareketi Eylem Ayarları ---
    SQUINT_THRESHOLD = 0.019 # Göz kısma eşiği
    BLINK_THRESHOLD = 0.012  # Göz kırpma eşiği (daha kapalı)
    ACTION_COOLDOWN = 0.8    # Eylemler arası genel bekleme süresi
    SQUINT_COOLDOWN = 0.4    # Göz kısarak sürekli zoom yapmak için bekleme süresi
    LONG_BLINK_DURATION = 0.6 # Uzun göz kırpmanın minimum süresi (saniye)
    last_action_time = 0
    last_squint_time = 0

    # Göz kırpma süresini ölçmek için durum değişkenleri
    is_in_blink = False
    blink_start_time = 0

    # --- Aktif Köşeler Ayarları (Devre Dışı) ---
    # CORNER_SIZE = 60
    # CORNER_DWELL_TIME = 1.0
    # HOT_CORNER_COOLDOWN = 3.0
    # last_hot_corner_time = 0

    print("Uygulama başlatılıyor, modeller yükleniyor...")
    model = tf.keras.models.load_model("mpiigaze_finetuned_v2.keras", compile=False)
    mp_face = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    
    # Yardımcı sınıfları başlat
    flow = GazeFlowManager()
    pupil = PupilTracker()
    intent_filter = GazeIntentFilter()
    smoother = TemporalSmoother(base_alpha=0.25)
    bias = BiasMap(grid=20)
    fix_logger = FixationLogger()
    
    L_EYE = [33,133,160,159,158,157,173]
    L_IRIS = [474,475,476,477]

    # Başlangıç ve durum değişkenleri
    nx, ny = 0.5, 0.5
    was_fixating = False
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Kamera açılamadı.")
        return

    print("✅ Uygulama başlatıldı. Çıkmak için 'ESC' tuşuna basın.")
    
    try:
        # --- Ana Döngü ---
        while cap.isOpened():
            learning_was_triggered_this_frame = False # Görsel geri bildirim için bayrak
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            res = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                cx = int(np.mean([lm[i].x * w for i in L_EYE]))
                cy = int(np.mean([lm[i].y * h for i in L_EYE]))
                roi = frame[cy-40:cy+40, cx-60:cx+60]

                if roi.size > 0:
                    current_time = time.time() # Zamanı döngünün başında bir kez al
                    pitch, yaw = model.predict(preprocess_eye(roi), verbose=0)[0]
                    
                    # Sinyal işleme ve birleştirme
                    nx_m, ny_m = flow.update(pitch, yaw)
                    dx, dy, mag = pupil.get([lm[i] for i in L_IRIS], w, h)
                    conf = intent_filter.confidence(nx_m, ny_m, mag)
                    
                    adaptive_gain = 0.20 + conf * 0.8
                    nx += (nx_m + dx * 3.2 - nx) * adaptive_gain
                    ny += (ny_m + dy * 3.2 - ny) * adaptive_gain
                    
                    nx_s, ny_s = smoother.update(nx, ny)
                    
                    # Odaklanma tespiti
                    fix_logger.update(nx_s, ny_s)
                    is_currently_fixating = fix_logger.is_fixating() and fix_logger.fixation_duration() > 0.2
                    
                    # Bias düzeltmesini uygula (imleç konumu için)
                    final_x_pre_clip, final_y_pre_clip = bias.apply(nx_s, ny_s)

                    # Aktif Öğrenme ve İmleç Kilidi Mantığı
                    if is_currently_fixating:
                        if not was_fixating: # Odaklanma yeni başladıysa
                            fix_x, fix_y = fix_logger.get_fixation_center()
                            
                            if ONLINE_LEARNING and conf > 0.5:
                                error_x = fix_x - nx_s
                                error_y = fix_y - ny_s
                                bias.learn(nx_s, ny_s, error_x, error_y, weight=conf * 0.5)
                                learning_was_triggered_this_frame = True
                            
                            # Odaklanma anındaki bias düzeltmesini al ve imleci kilitle
                            lock_x, lock_y = bias.apply(fix_x, fix_y)
                            final_x = np.clip(lock_x, 0, 1)
                            final_y = np.clip(lock_y, 0, 1)
                    else:
                        # Odaklanma yoksa, imleci serbestçe hareket ettir
                        final_x = np.clip(final_x_pre_clip, 0, 1)
                        final_y = np.clip(final_y_pre_clip, 0, 1)

                    pyautogui.moveTo(final_x * screen_w, final_y * screen_h, duration=0.1)
                    
                    scroll_progress = 0.0 # Kaydırma dolum çubuğu için ilerleme
                    # --- Akıcı Kaydırma Mantığı ---
                    if fix_logger.is_fixating():
                        # Odaklanma süresine göre kaydırma ilerlemesini hesapla
                        if fix_logger.fixation_duration() > 0: # Negatif duration'dan kaçın
                            scroll_progress = min(1.0, fix_logger.fixation_duration() / SCROLL_ACTIVATION_DWELL)

                        # Üst bölgeyi kontrol et (yukarı kaydırma)
                        if final_y * screen_h < SCROLL_ZONE_HEIGHT:
                            if scroll_progress >= 1.0 and (current_time - last_scroll_time) > SCROLL_COOLDOWN:
                                pyautogui.scroll(SCROLL_SPEED)
                                last_scroll_time = current_time
                        
                        # Alt bölgeyi kontrol et (aşağı kaydırma)
                        elif final_y * screen_h > screen_h - SCROLL_ZONE_HEIGHT:
                            if scroll_progress >= 1.0 and (current_time - last_scroll_time) > SCROLL_COOLDOWN:
                                pyautogui.scroll(-SCROLL_SPEED)
                                last_scroll_time = current_time

                    # --- Göz Hareketleri ile Eylem Mantığı ---
                    left_top = lm[159]
                    left_bottom = lm[145]
                    blink_ratio = left_bottom.y - left_top.y
                    
                    is_fully_closed = blink_ratio < BLINK_THRESHOLD
                    is_squinting = blink_ratio < SQUINT_THRESHOLD and not is_fully_closed

                    # 1. Göz Kısma (Yakınlaştırma) - Sürekli tetiklenebilir
                    if is_squinting:
                        if (current_time - last_squint_time) > SQUINT_COOLDOWN and (current_time - last_action_time) > SQUINT_COOLDOWN:
                            with pyautogui.hold('ctrl'):
                                pyautogui.scroll(120)
                            print(">>> SQUINT ZOOM IN <<<")
                            last_squint_time = current_time
                            last_action_time = current_time
                    
                    # 2. Göz Kırpma (Tıklama veya Uzaklaştırma) - Göz açıldığında tetiklenir
                    # Göz yeni kapandıysa, başlangıç zamanını kaydet
                    if is_fully_closed and not is_in_blink:
                        is_in_blink = True
                        blink_start_time = current_time
                    
                    # Göz yeni açıldıysa, süreyi hesapla ve eylemi gerçekleştir
                    elif not is_fully_closed and is_in_blink:
                        blink_duration = current_time - blink_start_time
                        is_in_blink = False

                        if (current_time - last_action_time) > ACTION_COOLDOWN:
                            # Uzun Göz Kırpma -> Uzaklaştırma
                            if blink_duration > LONG_BLINK_DURATION:
                                with pyautogui.hold('ctrl'):
                                    pyautogui.scroll(-120)
                                print(">>> LONG BLINK ZOOM OUT <<<")
                                last_action_time = current_time
                            # Kısa Göz Kırpma -> Tıklama
                            else:
                                pyautogui.click()
                                print(">>> CLICK <<<")
                                last_action_time = current_time

                    was_fixating = is_currently_fixating

                    # --- Görsel Geri Bildirim ---
                    # Odaklanma durumunu mavi bir halka ile görselleştir (Hata Ayıklama)
                    if fix_logger.is_fixating():
                        cv2.circle(frame, (cx, cy), 48, (255, 0, 0), 2) # Mavi halka

                    if learning_was_triggered_this_frame:
                        cv2.circle(frame, (cx, cy), 45, (0, 255, 0), 2) # Yeşil halka
                    
                    # Kaydırma dolum çubuğunu görselleştir
                    if scroll_progress > 0: # Sadece ilerleme varsa çiz
                        bar_height_px = int(scroll_progress * h) # Tam yükseklik
                        cv2.rectangle(frame, (0, h - bar_height_px), (15, h), (0, 255, 0), -1) # Sol altta yeşil dolan çubuk
                        cv2.rectangle(frame, (0, 0), (15, h), (100, 100, 100), 1) # Gri çerçeve

            cv2.imshow("Gaze Tracker - (Aktif Öğrenme) - Kamera (ESC)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
    except KeyboardInterrupt:
        print("ℹ️ Kullanıcı tarafından sonlandırıldı (CTRL+C).")
    finally:
        print("Uygulama kapatılıyor.")
        if ONLINE_LEARNING:
            bias.save()
            print("✅ Öğrenilen sapma (bias) ayarları kaydedildi.")
        cap.release()
        cv2.destroyAllWindows()

# =====================================================
if __name__ == "__main__":
    main()