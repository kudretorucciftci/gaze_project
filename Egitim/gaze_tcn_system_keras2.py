# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
from tensorflow.keras.layers import Input, Conv1D, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# =====================================================
# TCN MODEL (CPU-OPTIMAL, KERAS 2.x)
# =====================================================
def build_gaze_tcn(time_steps=12, features=3):
    inp = Input(shape=(time_steps, features))

    x = Conv1D(32, 3, padding="causal", dilation_rate=1, activation="relu")(inp)
    x = Conv1D(32, 3, padding="causal", dilation_rate=2, activation="relu")(x)
    x = Conv1D(32, 3, padding="causal", dilation_rate=4, activation="relu")(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)

    out = Dense(5)(x)
    # [gx, gy, vx, vy, stability]

    return Model(inp, out)

# =====================================================
# LOSS FUNCTION (STATE-BASED, STABLE)
# =====================================================
def gaze_state_loss(y_true, y_pred):
    pos = tf.reduce_mean(tf.square(y_true[:, :2] - y_pred[:, :2]))
    vel = tf.reduce_mean(tf.square(y_true[:, 2:4] - y_pred[:, 2:4]))
    stab = tf.reduce_mean(tf.square(y_true[:, 4] - y_pred[:, 4]))
    return pos + 0.3 * vel + 0.2 * stab

# =====================================================
# EYE PREPROCESS
# =====================================================
def preprocess_eye(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (60, 36)).astype(np.float32) / 255.0
    return img.reshape(1, 36, 60, 1)

# =====================================================
# TEMPORAL BUFFER
# =====================================================
class TemporalBuffer:
    def __init__(self, size=12):
        self.buf = deque(maxlen=size)

    def add(self, p, y, v):
        self.buf.append([p, y, v])

    def ready(self):
        return len(self.buf) == self.buf.maxlen

    def tensor(self):
        return np.expand_dims(np.array(self.buf, dtype=np.float32), axis=0)

# =====================================================
# SELF-SUPERVISED FLOW (TEACHER)
# =====================================================
class GazeFlow:
    def __init__(self):
        self.cx = 0.5
        self.cy = 0.5
        self.yaw_center = None

    def update(self, pitch, yaw):
        if self.yaw_center is None:
            self.yaw_center = yaw
        else:
            self.yaw_center += (yaw - self.yaw_center) * 0.01

        dx = yaw - self.yaw_center
        dy = -np.tanh(pitch * 1.7)

        tx = 0.5 - dx * 2.0
        ty = 0.5 + dy * 1.1

        self.cx += (tx - self.cx) * 0.25
        self.cy += (ty - self.cy) * 0.25

        return np.clip(self.cx, 0, 1), np.clip(self.cy, 0, 1)

# =====================================================
# IRIS VELOCITY
# =====================================================
class IrisVelocity:
    def __init__(self):
        self.last = None

    def compute(self, iris, w, h):
        pts = np.array([[p.x * w, p.y * h] for p in iris])
        c = pts.mean(axis=0)
        if self.last is None:
            self.last = c
            return 0.0
        v = np.linalg.norm(c - self.last)
        self.last = c
        return float(v)

# =====================================================
# MAIN TRAIN LOOP
# =====================================================
def main():
    TIME_STEPS = 12
    EPOCHS = 25
    TRAIN_SAMPLES = 200
    VAL_SAMPLES = 40

    gaze_cnn = tf.keras.models.load_model(
        "mpiigaze_fixed.keras", compile=False
    )

    tcn = build_gaze_tcn(TIME_STEPS, 3)
    tcn.compile(
        optimizer=Adam(1e-3),
        loss=gaze_state_loss
    )

    checkpoint = ModelCheckpoint(
        "gaze_tcn_best.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    mp_face = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    flow = GazeFlow()
    iris_vel = IrisVelocity()
    buffer = TemporalBuffer(TIME_STEPS)

    L_EYE = [33,133,160,159,158,157,173]
    L_IRIS = [474,475,476,477]

    cap = cv2.VideoCapture(0)

    def collect(n):
        X, Y = [], []
        prev_x, prev_y = 0.5, 0.5

        while len(X) < n:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            res = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                continue

            lm = res.multi_face_landmarks[0].landmark
            cx = int(np.mean([lm[i].x * w for i in L_EYE]))
            cy = int(np.mean([lm[i].y * h for i in L_EYE]))

            roi = frame[cy-40:cy+40, cx-60:cx+60]
            if roi.size == 0:
                continue

            pitch, yaw = gaze_cnn.predict(
                preprocess_eye(roi), verbose=0
            )[0]

            gx, gy = flow.update(pitch, yaw)
            iv = iris_vel.compute([lm[i] for i in L_IRIS], w, h)

            buffer.add(pitch, yaw, iv)

            if buffer.ready():
                vx = gx - prev_x
                vy = gy - prev_y
                stability = 1.0 - min(np.sqrt(vx*vx + vy*vy) * 25, 1.0)

                X.append(buffer.tensor()[0])
                Y.append([gx, gy, vx, vy, stability])

                prev_x, prev_y = gx, gy

        return np.array(X), np.array(Y)

    print("🎓 CPU TCN TRAINING BAŞLADI")

    for ep in range(EPOCHS):
        Xtr, Ytr = collect(TRAIN_SAMPLES)
        Xv, Yv = collect(VAL_SAMPLES)

        hist = tcn.fit(
            Xtr, Ytr,
            validation_data=(Xv, Yv),
            epochs=1,
            batch_size=32,
            callbacks=[checkpoint],
            verbose=1
        )

        print(
            f"Epoch {ep+1}/{EPOCHS} | "
            f"loss={hist.history['loss'][0]:.4f} | "
            f"val_loss={hist.history['val_loss'][0]:.4f}"
        )

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Eğitim tamamlandı. En iyi model diskte: gaze_tcn_best.keras")

# =====================================================
if __name__ == "__main__":
    main()
