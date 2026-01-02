# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
from gaze_a3_a4_a5 import TemporalSmoother, BiasMap, IntentDetector
from fixation_logger import FixationLogger

ONLINE_LEARNING = True
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# =====================================================
class DirectionalErrorVoter:
    def __init__(self, maxlen=12, max_corr=0.035):
        self.buf = deque(maxlen=maxlen)
        self.max_corr = max_corr

    def update(self, err_x, err_y, allow, intent):
        if allow and intent > 0.55:
            self.buf.append((err_x, err_y))
        elif len(self.buf) > 0:
            self.buf.popleft()

    def correction(self, intent):
        if len(self.buf) < 3:
            return 0.0, 0.0

        ex = np.mean([e[0] for e in self.buf])
        ey = np.mean([e[1] for e in self.buf])

        mag = np.sqrt(ex * ex + ey * ey)
        if mag < 1e-4:
            return 0.0, 0.0

        scale = intent ** 1.5
        cx = np.clip(ex * scale, -self.max_corr, self.max_corr)
        cy = np.clip(ey * scale, -self.max_corr, self.max_corr)

        return cx, cy

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
        r = np.sqrt(dx * dx + dy * dy)

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
class AdaptiveGazeBox:
    def __init__(self, lr=0.02):
        self.xmin = 0.45
        self.xmax = 0.55
        self.ymin = 0.45
        self.ymax = 0.55
        self.lr = lr

    def update(self, nx, ny, allow):
        if not allow:
            self.xmin += self.lr * (min(self.xmin, nx) - self.xmin)
            self.xmax += self.lr * (max(self.xmax, nx) - self.xmax)
            self.ymin += self.lr * (min(self.ymin, ny) - self.ymin)
            self.ymax += self.lr * (max(self.ymax, ny) - self.ymax)

        self.xmin = np.clip(self.xmin, 0.0, 0.9)
        self.xmax = np.clip(self.xmax, 0.1, 1.0)
        self.ymin = np.clip(self.ymin, 0.0, 0.9)
        self.ymax = np.clip(self.ymax, 0.1, 1.0)

    def remap(self, nx, ny):
        x = (nx - self.xmin) / max(1e-4, (self.xmax - self.xmin))
        y = (ny - self.ymin) / max(1e-4, (self.ymax - self.ymin))
        return np.clip(x, 0, 1), np.clip(y, 0, 1)

    def draw(self, frame, w, h):
        cv2.rectangle(
            frame,
            (int(self.xmin * w), int(self.ymin * h)),
            (int(self.xmax * w), int(self.ymax * h)),
            (0, 255, 255),
            2
        )

# =====================================================
def preprocess_eye(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (60, 36)).astype(np.float32) / 255.0
    return img[None, ..., None]

# =====================================================
def main():
    model = tf.keras.models.load_model("mpiigaze_fixed.keras", compile=False)
    mp_face = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("❌ Kamera açılamadı")
        return

    flow = GazeFlowManager()
    pupil = PupilTracker()
    intent_filter = GazeIntentFilter()
    smoother = TemporalSmoother(base_alpha=0.50)
    bias = BiasMap(grid=20)
    fix_logger = FixationLogger()
    gaze_box = AdaptiveGazeBox()
    error_voter = DirectionalErrorVoter()

    L_EYE = [33, 133, 160, 159, 158, 157, 173]
    L_IRIS = [474, 475, 476, 477]

    nx, ny = 0.5, 0.5
    nx_vis, ny_vis = 0.5, 0.5

    fixed_gaze_x, fixed_gaze_y = 0.5, 0.5
    is_gaze_fixed = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        res = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            cx = int(np.mean([lm[i].x * w for i in L_EYE]))
            cy = int(np.mean([lm[i].y * h for i in L_EYE]))

            roi = frame[max(0, cy-40):min(h, cy+40),
                        max(0, cx-60):min(w, cx+60)]

            if roi.size > 0:
                pitch, yaw = model.predict(preprocess_eye(roi), verbose=0)[0]

                nx_m, ny_m = flow.update(pitch, yaw)
                dx, dy, mag = pupil.get([lm[i] for i in L_IRIS], w, h)
                conf = intent_filter.confidence(nx_m, ny_m, mag)

                nx += (nx_m + dx * 3.2 - nx) * (0.20 + conf * 0.8)
                ny += (ny_m + dy * 3.2 - ny) * (0.20 + conf * 0.8)

                nx, ny = smoother.update(nx, ny)
                nx_b, ny_b = bias.apply(nx, ny)
                fix_logger.update(nx_b, ny_b)

                allow = conf > 0.45 and fix_logger.fixation_duration() > 0.15
                gaze_box.update(nx_b, ny_b, allow)

                if allow and not is_gaze_fixed:
                    fixed_gaze_x, fixed_gaze_y = gaze_box.remap(nx_b, ny_b)
                    is_gaze_fixed = True
                elif not allow:
                    is_gaze_fixed = False

                display_nx_s, display_ny_s = (
                    (fixed_gaze_x, fixed_gaze_y) if is_gaze_fixed else (nx, ny)
                )

                err_x = display_nx_s - nx
                err_y = display_ny_s - ny
                error_voter.update(err_x, err_y, allow, conf)
                corr_x, corr_y = error_voter.correction(conf)

                display_nx_s = np.clip(display_nx_s + corr_x, 0, 1)
                display_ny_s = np.clip(display_ny_s + corr_y, 0, 1)

                if ONLINE_LEARNING and allow and is_gaze_fixed and conf > 0.6:
                    bias.learn(nx, ny, err_x, err_y, weight=conf)

                cv2.circle(
                    frame,
                    (int(display_nx_s * w), int(display_ny_s * h)),
                    8,
                    (0, 255, 0) if allow else (0, 0, 255),
                    -1
                )

                gaze_box.draw(frame, w, h)

        cv2.imshow("GAZE — ADIM 4", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    bias.save()
    cap.release()
    cv2.destroyAllWindows()

# =====================================================
if __name__ == "__main__":
    main()
