# -*- coding: utf-8 -*-
"""
FAST SNAP GAZE SYSTEM (ORIGINAL)
+ OPTIONAL IMPLICIT SELF-CALIBRATION
+ OPTIONAL PHONE PORTRAIT ORIENTATION FIX

• Eski davranış %100 korunur
• Snap / hız / gain değişmez
• Ek özellikler tak-çıkar
"""

import os
import cv2
import time
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ================== KONTROLLER ==================
ENABLE_IMPLICIT_CALIB = True
ROTATE_PHONE_PORTRAIT = False     # telefonu dikey tutuyorsan True
PROFILE_PATH = "implicit_bias.npz"
# ================================================

# =====================================================
# GAZE FLOW (ESKİ – DOKUNULMADI)
# =====================================================
class GazeFlowManager:
    def __init__(self):
        self.nx = 0.5
        self.ny = 0.5
        self.yaw_center = 0.175
        self.lin_gain_x = 2.0
        self.lin_gain_y = 1.5
        self.non_gain_x = 55.0
        self.non_gain_y = 28.0
        self.interp = 0.22

    def update(self, pitch, yaw):
        dx = yaw - self.yaw_center
        dy = pitch
        r = np.sqrt(dx*dx + dy*dy)

        tx_lin = 0.5 - dx*self.lin_gain_x
        ty_lin = 0.5 + dy*self.lin_gain_y

        mx = np.sign(dx)*(abs(dx)**0.75)*self.non_gain_x
        my = np.sign(dy)*(abs(dy)**0.75)*self.non_gain_y

        tx_non = 0.5 - mx
        ty_non = 0.5 + my

        w = np.clip((r-0.04)/0.12,0,1)
        tx = (1-w)*tx_lin + w*tx_non
        ty = (1-w)*ty_lin + w*ty_non

        self.nx += (tx-self.nx)*self.interp
        self.ny += (ty-self.ny)*self.interp
        return np.clip(self.nx,0,1), np.clip(self.ny,0,1)

# =====================================================
# PUPIL TRACKER (ESKİ)
# =====================================================
class PupilTracker:
    def __init__(self):
        self.last = None
        self.smooth = 0.65

    def get(self, iris, w, h):
        pts = np.array([[p.x*w,p.y*h] for p in iris])
        c = pts.mean(axis=0)
        if self.last is None:
            self.last = c
            return 0,0,0
        delta = c-self.last
        self.last = self.last*self.smooth + c*(1-self.smooth)
        return delta[0]/w, delta[1]/h, np.linalg.norm(delta)

# =====================================================
# INTENT FILTER (ESKİ SNAP DAVRANIŞI)
# =====================================================
class GazeIntentFilter:
    def __init__(self):
        self.hist = deque(maxlen=10)
        self.intent = 0.0

    def confidence(self, nx, ny, mag):
        self.hist.append((nx,ny))
        if len(self.hist)<5:
            self.intent*=0.8
            return self.intent
        xs=[p[0] for p in self.hist]
        ys=[p[1] for p in self.hist]
        std=np.sqrt(np.var(xs)+np.var(ys))
        spatial=np.clip(1-std*130,0,1)
        pupil=np.exp(-((mag-0.003)**2)/0.000015)
        raw=0.6*spatial+0.4*pupil
        if raw>self.intent:
            self.intent+=(raw-self.intent)*0.35
        else:
            self.intent*=0.75
        return np.clip(self.intent,0,1)

# =====================================================
# 🔹 YENİ: PASİF ÖRTÜK KALİBRASYON (DAVRANIŞI BOZMAZ)
# =====================================================
class ImplicitBiasMap:
    def __init__(self, grid=20):
        self.g=grid
        self.bx=np.zeros((grid,grid))
        self.by=np.zeros((grid,grid))
        self.lr_fast=0.015
        self.lr_slow=0.002
        self.maxb=0.12
        if os.path.exists(PROFILE_PATH):
            d=np.load(PROFILE_PATH)
            self.bx=d["bx"]; self.by=d["by"]

    def cell(self,x,y):
        return int(np.clip(x*self.g,0,self.g-1)),int(np.clip(y*self.g,0,self.g-1))

    def apply(self,x,y):
        ix,iy=self.cell(x,y)
        return np.clip(x+self.bx[iy,ix],0,1),np.clip(y+self.by[iy,ix],0,1)

    def update(self,x,y,dx,dy,fast):
        if abs(dx)+abs(dy)<0.002: return
        ix,iy=self.cell(x,y)
        lr=self.lr_fast if fast else self.lr_slow
        self.bx[iy,ix]=np.clip(self.bx[iy,ix]+dx*lr,-self.maxb,self.maxb)
        self.by[iy,ix]=np.clip(self.by[iy,ix]+dy*lr,-self.maxb,self.maxb)

    def save(self):
        np.savez(PROFILE_PATH,bx=self.bx,by=self.by)

# =====================================================
def preprocess_eye(roi):
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    img=cv2.resize(gray,(60,36)).astype(np.float32)/255.0
    return img[None,...,None]

# =====================================================
def main():
    model=tf.keras.models.load_model("mpiigaze_fixed.keras",compile=False)
    mp_face=mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    cap=cv2.VideoCapture(0)

    flow=GazeFlowManager()
    pupil=PupilTracker()
    intent=GazeIntentFilter()
    bias=ImplicitBiasMap()

    L_EYE=[33,133,160,159,158,157,173]
    L_IRIS=[474,475,476,477]

    nx,ny=0.5,0.5
    t0=time.time()

    while cap.isOpened():
        ret,frame=cap.read()
        if not ret: break

        # if ROTATE_PHONE_PORTRAIT:
        #     frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        frame = cv2.flip(frame,1)
        h,w=frame.shape[:2]

        res=mp_face.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            lm=res.multi_face_landmarks[0].landmark
            cx=int(np.mean([lm[i].x*w for i in L_EYE]))
            cy=int(np.mean([lm[i].y*h for i in L_EYE]))
            roi=frame[cy-40:cy+40,cx-60:cx+60]

            if roi.size>0:
                pitch,yaw=model.predict(preprocess_eye(roi),verbose=0)[0]
                nx_m,ny_m=flow.update(pitch,yaw)
                dx,dy,mag=pupil.get([lm[i] for i in L_IRIS],w,h)
                conf=intent.confidence(nx_m,ny_m,mag)

                adaptive_gain=0.10+conf*0.65
                nx+=(nx_m+dx*2.2-nx)*adaptive_gain
                ny+=(ny_m+dy*2.2-ny)*adaptive_gain

                if ENABLE_IMPLICIT_CALIB:
                    fast=(time.time()-t0)<10
                    bias.update(nx,ny,dx,dy,fast)
                    nx,ny=bias.apply(nx,ny)

                cv2.circle(frame,(int(nx*w),int(ny*h)),8,(0,0,255),-1)

        cv2.imshow("FAST GAZE (UNCHANGED) + EXTENSIONS",frame)
        if cv2.waitKey(1)&0xFF==27: break

    if ENABLE_IMPLICIT_CALIB:
        bias.save()

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()

    # ---------------------------------------------------2. ana --------------------- 
 # -*- coding: utf-8 -*-
# import os
# import cv2
# import numpy as np
# import mediapipe as mp
# import tensorflow as tf
# from collections import deque
# from gaze_a3_a4_a5 import TemporalSmoother, BiasMap, IntentDetector
# from fixation_logger import FixationLogger

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# ROTATE_PHONE_PORTRAIT = True

# # =====================================================
# class GazeFlowManager:
#     def __init__(self):
#         self.nx = 0.5
#         self.ny = 0.5
#         self.yaw_center = 0.175
#         self.lin_gain_x = 2.0
#         self.lin_gain_y = 1.5
#         self.non_gain_x = 55.0
#         self.non_gain_y = 28.0
#         self.interp = 0.22

#     def update(self, pitch, yaw):
#         dx = yaw - self.yaw_center
#         dy = pitch
#         r = np.sqrt(dx*dx + dy*dy)
#         tx_lin = 0.5 - dx*self.lin_gain_x
#         ty_lin = 0.5 + dy*self.lin_gain_y
#         mx = np.sign(dx)*(abs(dx)**0.75)*self.non_gain_x
#         my = np.sign(dy)*(abs(dy)**0.75)*self.non_gain_y
#         tx_non = 0.5 - mx
#         ty_non = 0.5 + my
#         w = np.clip((r-0.04)/0.12,0,1)
#         tx = (1-w)*tx_lin + w*tx_non
#         ty = (1-w)*ty_lin + w*ty_non
#         self.nx += (tx-self.nx)*self.interp
#         self.ny += (ty-self.ny)*self.interp
#         return np.clip(self.nx,0,1), np.clip(self.ny,0,1)

# # =====================================================
# class PupilTracker:
#     def __init__(self):
#         self.last = None
#         self.smooth = 0.65

#     def get(self, iris, w, h):
#         pts = np.array([[p.x*w,p.y*h] for p in iris])
#         c = pts.mean(axis=0)
#         if self.last is None:
#             self.last = c
#             return 0,0,0
#         delta = c-self.last
#         self.last = self.last*self.smooth + c*(1-self.smooth)
#         return delta[0]/w, delta[1]/h, np.linalg.norm(delta)

# # =====================================================
# class GazeIntentFilter:
#     def __init__(self):
#         self.hist = deque(maxlen=10)
#         self.intent = 0.0

#     def confidence(self, nx, ny, mag):
#         self.hist.append((nx,ny))
#         if len(self.hist)<5:
#             self.intent*=0.8
#             return self.intent
#         xs=[p[0] for p in self.hist]
#         ys=[p[1] for p in self.hist]
#         std=np.sqrt(np.var(xs)+np.var(ys))
#         spatial=np.clip(1-std*130,0,1)
#         pupil=np.exp(-((mag-0.003)**2)/0.000015)
#         raw=0.6*spatial+0.4*pupil
#         if raw>self.intent:
#             self.intent+=(raw-self.intent)*0.35
#         else:
#             self.intent*=0.75
#         return np.clip(self.intent,0,1)

# # =====================================================
# def preprocess_eye(roi):
#     gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
#     img=cv2.resize(gray,(60,36)).astype(np.float32)/255.0
#     return img[None,...,None]

# # =====================================================
# def main():
#     model=tf.keras.models.load_model("mpiigaze_fixed.keras",compile=False)
#     mp_face=mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
#     cap=cv2.VideoCapture(0)

#     flow=GazeFlowManager()
#     pupil=PupilTracker()
#     intent_filter=GazeIntentFilter()

#     smoother = TemporalSmoother(base_alpha=0.20)
#     bias = BiasMap(grid=20)
#     intent = IntentDetector()
#     fix_logger = FixationLogger()

#     L_EYE=[33,133,160,159,158,157,173]
#     L_IRIS=[474,475,476,477]

#     nx,ny=0.5,0.5

#     while cap.isOpened():
#         ret,frame=cap.read()
#         if not ret: break

#         if ROTATE_PHONE_PORTRAIT:
#             frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
#         frame = cv2.flip(frame,1)
#         h,w=frame.shape[:2]
#         nx = 1.0 - nx  # x eksenini tersle


#         res=mp_face.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
#         if res.multi_face_landmarks:
#             lm=res.multi_face_landmarks[0].landmark
#             cx=int(np.mean([lm[i].x*w for i in L_EYE]))
#             cy=int(np.mean([lm[i].y*h for i in L_EYE]))
#             roi=frame[cy-40:cy+40,cx-60:cx+60]

#             if roi.size>0:
#                 pitch,yaw=model.predict(preprocess_eye(roi),verbose=0)[0]
#                 nx_m,ny_m=flow.update(pitch,yaw)

#                 dx,dy,mag=pupil.get([lm[i] for i in L_IRIS],w,h)
#                 conf=intent_filter.confidence(nx_m,ny_m,mag)

#                 adaptive_gain=0.10+conf*0.65
#                 nx+=(nx_m+dx*2.2-nx)*adaptive_gain
#                 ny+=(ny_m+dy*2.2-ny)*adaptive_gain

#                 nx, ny = smoother.update(nx, ny)
#                 nx_b, ny_b = bias.apply(nx, ny)

#                 # Mikro snap
#                 if intent.update(nx_b, ny_b):
#                     bias.learn(nx, ny, nx_b-nx, ny_b-ny)
#                     snap_strength = 0.03
#                     nx += (nx_b - nx) * snap_strength
#                     ny += (ny_b - ny) * snap_strength

#                 # Ekran kenarı stabilizasyonu
#                 edge_margin = 0.03
#                 edge_strength = 0.4
#                 if nx < edge_margin:
#                     nx += (edge_margin - nx) * edge_strength
#                 elif nx > 1 - edge_margin:
#                     nx -= (nx - (1 - edge_margin)) * edge_strength
#                 if ny < edge_margin:
#                     ny += (edge_margin - ny) * edge_strength
#                 elif ny > 1 - edge_margin:
#                     ny -= (ny - (1 - edge_margin)) * edge_strength

#                 nx, ny = nx_b, ny_b  # bias sonrası son değer

#                 # Fixation log
#                 fix_logger.update(nx, ny)

#                 cv2.circle(frame,(int(nx*w),int(ny*h)),8,(0,0,255),-1)

#         cv2.imshow("FAST GAZE + SNAP + EDGE + FIX",frame)
#         if cv2.waitKey(1)&0xFF==27:
#             break

#     bias.save()
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__=="__main__":
#     main()



# # -*- coding: utf-8 -*-
# import os
# import cv2
# import numpy as np
# import mediapipe as mp
# import tensorflow as tf
# from collections import deque
# from gaze_a3_a4_a5 import TemporalSmoother, BiasMap, IntentDetector
# from fixation_logger import FixationLogger

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# ROTATE_PHONE_PORTRAIT = False

# # =====================================================
# class GazeFlowManager:
#     def __init__(self):
#         self.nx = 0.5
#         self.ny = 0.5
#         self.yaw_center = 0.175
#         self.lin_gain_x = 2.0
#         self.lin_gain_y = 1.5
#         self.non_gain_x = 55.0
#         self.non_gain_y = 28.0
#         self.interp = 0.22

#     def update(self, pitch, yaw):
#         dx = yaw - self.yaw_center
#         dy = pitch
#         r = np.sqrt(dx*dx + dy*dy)

#         tx_lin = 0.5 - dx*self.lin_gain_x
#         ty_lin = 0.5 + dy*self.lin_gain_y

#         mx = np.sign(dx)*(abs(dx)**0.75)*self.non_gain_x
#         my = np.sign(dy)*(abs(dy)**0.75)*self.non_gain_y
#         tx_non = 0.5 - mx
#         ty_non = 0.5 + my

#         w = np.clip((r-0.04)/0.12,0,1)
#         tx = (1-w)*tx_lin + w*tx_non
#         ty = (1-w)*ty_lin + w*ty_non

#         self.nx += (tx-self.nx)*self.interp
#         self.ny += (ty-self.ny)*self.interp
#         return np.clip(self.nx,0,1), np.clip(self.ny,0,1)


# # =====================================================
# class PupilTracker:
#     def __init__(self):
#         self.last = None
#         self.smooth = 0.65

#     def get(self, iris, w, h):
#         pts = np.array([[p.x*w,p.y*h] for p in iris])
#         c = pts.mean(axis=0)
#         if self.last is None:
#             self.last = c
#             return 0,0,0
#         delta = c-self.last
#         self.last = self.last*self.smooth + c*(1-self.smooth)
#         return delta[0]/w, delta[1]/h, np.linalg.norm(delta)


# # =====================================================
# class GazeIntentFilter:
#     def __init__(self):
#         self.hist = deque(maxlen=10)
#         self.intent = 0.0

#     def confidence(self, nx, ny, mag):
#         self.hist.append((nx,ny))
#         if len(self.hist)<5:
#             self.intent*=0.8
#             return self.intent

#         xs=[p[0] for p in self.hist]
#         ys=[p[1] for p in self.hist]
#         std=np.sqrt(np.var(xs)+np.var(ys))

#         spatial=np.clip(1-std*130,0,1)
#         pupil=np.exp(-((mag-0.003)**2)/0.000015)

#         raw=0.6*spatial+0.4*pupil

#         if raw>self.intent:
#             self.intent+=(raw-self.intent)*0.35
#         else:
#             self.intent*=0.75

#         return np.clip(self.intent,0,1)


# # =====================================================
# def preprocess_eye(roi):
#     gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
#     img=cv2.resize(gray,(60,36)).astype(np.float32)/255.0
#     return img[None,...,None]


# # =====================================================
# def main():
#     model=tf.keras.models.load_model(
#         "mpiigaze_finetuned.keras", compile=False
#     )

#     mp_face=mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
#     cap=cv2.VideoCapture(0)

#     flow=GazeFlowManager()
#     pupil=PupilTracker()
#     intent_filter=GazeIntentFilter()

#     smoother = TemporalSmoother(base_alpha=0.20)
#     bias = BiasMap(grid=20)
#     intent = IntentDetector()
#     fix_logger = FixationLogger()

#     L_EYE=[33,133,160,159,158,157,173]
#     L_IRIS=[474,475,476,477]

#     nx,ny=0.5,0.5

#     while cap.isOpened():
#         ret,frame=cap.read()
#         if not ret: break

#         # if ROTATE_PHONE_PORTRAIT:
#         #     frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
#         frame = cv2.flip(frame,1)

#         h,w=frame.shape[:2]
#         nx = 1.0 - nx

#         res=mp_face.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
#         if res.multi_face_landmarks:
#             lm=res.multi_face_landmarks[0].landmark

#             cx=int(np.mean([lm[i].x*w for i in L_EYE]))
#             cy=int(np.mean([lm[i].y*h for i in L_EYE]))
#             roi=frame[cy-40:cy+40,cx-60:cx+60]

#             if roi.size>0:
#                 pitch,yaw=model.predict(preprocess_eye(roi),verbose=0)[0]
#                 nx_m,ny_m=flow.update(pitch,yaw)

#                 dx,dy,mag=pupil.get([lm[i] for i in L_IRIS],w,h)
#                 conf=intent_filter.confidence(nx_m,ny_m,mag)

#                 adaptive_gain=0.10+conf*0.65
#                 nx+=(nx_m+dx*2.2-nx)*adaptive_gain
#                 ny+=(ny_m+dy*2.2-ny)*adaptive_gain

#                 nx, ny = smoother.update(nx, ny)
#                 nx_b, ny_b = bias.apply(nx, ny)

#                 fix_logger.update(nx_b, ny_b)

#                 # ===== ADIM 3: ZAMAN AĞIRLIKLI ÖDÜL =====
#                 conf_w = np.clip((conf - 0.6) / 0.4, 0, 1) ** 1.5
#                 fix_t = fix_logger.fixation_duration()
#                 fix_w = np.clip((fix_t - 0.25) / 1.25, 0, 1) ** 1.2
#                 final_w = conf_w * fix_w

#                 reward_ok = (
#                     conf > 0.6 and
#                     fix_logger.is_fixating() and
#                     smoother.stable()
#                 )

#                 if reward_ok and intent.update(nx_b, ny_b):
#                     bias.learn(
#                         nx, ny,
#                         nx_b-nx,
#                         ny_b-ny,
#                         weight=final_w
#                     )

#                     snap = 0.03 * final_w
#                     nx += (nx_b-nx)*snap
#                     ny += (ny_b-ny)*snap

#                 nx, ny = nx_b, ny_b
#                 cv2.circle(frame,(int(nx*w),int(ny*h)),8,(0,0,255),-1)

#         cv2.imshow("GAZE — ADIM 3",frame)
#         if cv2.waitKey(1)&0xFF==27:
#             break

#     bias.save()
#     cap.release()
#     cv2.destroyAllWindows()


# if __name__=="__main__":
#     main()