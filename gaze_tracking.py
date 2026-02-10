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

from gaze_utils import TemporalSmoother, BiasMap, IntentDetector
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
    # --- Setup ---
    pyautogui.FAILSAFE = False
    screen_w, screen_h = pyautogui.size()
    print(f"Screen Resolution: {screen_w}x{screen_h}")

    # --- Smooth Scrolling Settings ---
    SCROLL_ZONE_HEIGHT = 70
    SCROLL_ACTIVATION_DWELL = 0.6
    SCROLL_SPEED = 40
    SCROLL_COOLDOWN = 0.05
    last_scroll_time = 0

    # --- Eye Gesture Action Settings ---
    SQUINT_THRESHOLD = 0.019
    BLINK_THRESHOLD = 0.012
    ACTION_COOLDOWN = 0.8
    SQUINT_COOLDOWN = 0.4
    LONG_BLINK_DURATION = 0.5
    last_action_time = 0
    last_squint_time = 0

    # Variables for blink and state tracking
    is_left_eye_closed_state = False
    is_right_eye_closed_state = False
    left_eye_closed_timestamp = 0.0
    right_eye_closed_timestamp = 0.0

    print("Initializing application, loading models...")
    model = tf.keras.models.load_model("mpiigaze_finetuned_v2.keras", compile=False)
    mp_face = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    
    # Initialize helper classes
    flow = GazeFlowManager()
    pupil = PupilTracker()
    intent_filter = GazeIntentFilter()
    smoother = TemporalSmoother(base_alpha=0.25)
    bias = BiasMap(grid=20)
    fix_logger = FixationLogger()
    
    L_EYE = [33,133,160,159,158,157,173]
    R_EYE = [362,382,381,380,374,373,398]
    L_IRIS = [474,475,476,477]

    LEFT_EYE_TOP_LM = 159
    LEFT_EYE_BOTTOM_LM = 145
    RIGHT_EYE_TOP_LM = 386 
    RIGHT_EYE_BOTTOM_LM = 374 

    nx, ny = 0.5, 0.5
    was_fixating = False
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open camera.")
        return

    print("✅ Application started. Press 'ESC' to exit.")
    
    try:
        while cap.isOpened():
            learning_was_triggered_this_frame = False
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            res = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                
                cx_left = int(np.mean([lm[i].x * w for i in L_EYE]))
                cy_left = int(np.mean([lm[i].y * h for i in L_EYE]))
                roi_left = frame[cy_left-40:cy_left+40, cx_left-60:cx_left+60]
                
                cx = int(np.mean([lm[i].x * w for i in L_EYE + R_EYE]))
                cy = int(np.mean([lm[i].y * h for i in L_EYE + R_EYE]))
                
                if roi_left.size > 0:
                    current_time = time.time()
                    pitch, yaw = model.predict(preprocess_eye(roi_left), verbose=0)[0]
                    
                    nx_m, ny_m = flow.update(pitch, yaw)
                    dx, dy, mag = pupil.get([lm[i] for i in L_IRIS], w, h)
                    conf = intent_filter.confidence(nx_m, ny_m, mag)
                    
                    adaptive_gain = 0.20 + conf * 0.8
                    nx += (nx_m + dx * 3.2 - nx) * adaptive_gain
                    ny += (ny_m + dy * 3.2 - ny) * adaptive_gain
                    
                    nx_s, ny_s = smoother.update(nx, ny)
                    
                    fix_logger.update(nx_s, ny_s)
                    is_currently_fixating = fix_logger.is_fixating() and fix_logger.fixation_duration() > 0.2
                    
                    final_x_pre_clip, final_y_pre_clip = bias.apply(nx_s, ny_s)

                    if is_currently_fixating:
                        if not was_fixating:
                            fix_x, fix_y = fix_logger.get_fixation_center()
                            
                            if ONLINE_LEARNING and conf > 0.5:
                                error_x = fix_x - nx_s
                                error_y = fix_y - ny_s
                                # Dynamic Weighted Learning
                                bias.learn(nx_s, ny_s, error_x, error_y, fix_logger.fixation_duration(), conf)
                                learning_was_triggered_this_frame = True
                            
                            lock_x, lock_y = bias.apply(fix_x, fix_y)
                            final_x = np.clip(lock_x, 0, 1)
                            final_y = np.clip(lock_y, 0, 1)
                    else:
                        final_x = np.clip(final_x_pre_clip, 0, 1)
                        final_y = np.clip(final_y_pre_clip, 0, 1)

                    pyautogui.moveTo(final_x * screen_w, final_y * screen_h)
                    
                    scroll_progress = 0.0
                    if fix_logger.is_fixating():
                        if fix_logger.fixation_duration() > 0:
                            scroll_progress = min(1.0, fix_logger.fixation_duration() / SCROLL_ACTIVATION_DWELL)

                        if final_y * screen_h < SCROLL_ZONE_HEIGHT:
                            if scroll_progress >= 1.0 and (current_time - last_scroll_time) > SCROLL_COOLDOWN:
                                pyautogui.scroll(SCROLL_SPEED)
                                last_scroll_time = current_time
                        
                        elif final_y * screen_h > screen_h - SCROLL_ZONE_HEIGHT:
                            if scroll_progress >= 1.0 and (current_time - last_scroll_time) > SCROLL_COOLDOWN:
                                pyautogui.scroll(-SCROLL_SPEED)
                                last_scroll_time = current_time

                    # --- Action Logic with Eye Gestures (WITHOUT Reward/Penalty) ---
                    left_eye_ratio = lm[LEFT_EYE_BOTTOM_LM].y - lm[LEFT_EYE_TOP_LM].y
                    right_eye_ratio = lm[RIGHT_EYE_BOTTOM_LM].y - lm[RIGHT_EYE_TOP_LM].y

                    is_left_closed_now = left_eye_ratio < BLINK_THRESHOLD
                    is_right_closed_now = right_eye_ratio < BLINK_THRESHOLD
                    is_squinting_now = (left_eye_ratio < SQUINT_THRESHOLD and not is_left_closed_now) or \
                                     (right_eye_ratio < SQUINT_THRESHOLD and not is_right_closed_now)
                    
                    action_taken_this_frame = False
                    
                    if (current_time - last_action_time) > ACTION_COOLDOWN:
                        # 1. Long Bilateral Blink (Zoom Out)
                        if is_left_closed_now and is_right_closed_now:
                            if not is_left_eye_closed_state: left_eye_closed_timestamp = current_time
                            is_left_eye_closed_state = True
                            
                            if (current_time - left_eye_closed_timestamp) > LONG_BLINK_DURATION:
                                with pyautogui.hold('ctrl'):
                                    pyautogui.scroll(-120)
                                print(">>> LONG BLINK ZOOM OUT <<<")
                                last_action_time = current_time
                                action_taken_this_frame = True
                                is_left_eye_closed_state = False
                        
                        # 2. Squinting (Zoom In)
                        elif not action_taken_this_frame and is_squinting_now:
                            if (current_time - last_squint_time) > SQUINT_COOLDOWN:
                                with pyautogui.hold('ctrl'):
                                    pyautogui.scroll(120)
                                print(">>> SQUINT ZOOM IN <<<")
                                last_squint_time, last_action_time = current_time, current_time
                                action_taken_this_frame = True

                    # 3. Short Blinks (Right/Left Click)
                    if not is_left_closed_now and is_left_eye_closed_state:
                        if not action_taken_this_frame and not is_right_eye_closed_state:
                            if (current_time - last_action_time) > ACTION_COOLDOWN:
                                pyautogui.click(button='left')
                                print(">>> LEFT CLICK (Bilateral/Short) <<<")
                                last_action_time = current_time
                        is_left_eye_closed_state = False
                    
                    if not is_right_closed_now and is_right_eye_closed_state:
                        if not action_taken_this_frame and not is_left_eye_closed_state:
                             if (current_time - last_action_time) > ACTION_COOLDOWN:
                                pyautogui.click(button='right')
                                print(">>> RIGHT CLICK (Unilateral) <<<")
                                last_action_time = current_time
                        is_right_eye_closed_state = False

                    # Update eye closed states
                    if is_left_closed_now and not is_left_eye_closed_state:
                        is_left_eye_closed_state = True
                        left_eye_closed_timestamp = current_time
                    if is_right_closed_now and not is_right_eye_closed_state:
                        is_right_eye_closed_state = True
                        right_eye_closed_timestamp = current_time
                    
                    was_fixating = is_currently_fixating

                    if fix_logger.is_fixating():
                        cv2.circle(frame, (cx, cy), 48, (255, 0, 0), 2)
                    
                    if learning_was_triggered_this_frame:
                        cv2.circle(frame, (cx, cy), 45, (0, 255, 0), 2)
                    
                    if scroll_progress > 0:
                        bar_height_px = int(scroll_progress * h)
                        cv2.rectangle(frame, (0, h - bar_height_px), (15, h), (0, 255, 0), -1)
                        cv2.rectangle(frame, (0, 0), (15, h), (100, 100, 100), 1)

            cv2.imshow("Gaze Tracker - (Active Learning) - Camera (ESC)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
    except KeyboardInterrupt:
        print("ℹ️ Terminated by user (CTRL+C).")
    finally:
        print("Closing application.")
        if ONLINE_LEARNING:
            bias.save()
            print("✅ Learned bias settings saved.")
        cap.release()
        cv2.destroyAllWindows()

# =====================================================
if __name__ == "__main__":
    main()
