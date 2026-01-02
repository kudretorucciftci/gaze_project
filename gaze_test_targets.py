# -*- coding: utf-8 -*-
import cv2
import math

class GazeTestTargets:
    def __init__(self):
        # Normalized target positions (0–1)
        self.targets = [
            {"pos": (0.5, 0.35), "hit": False},
            {"pos": (0.35, 0.5), "hit": False},
            {"pos": (0.65, 0.5), "hit": False},
            {"pos": (0.5, 0.65), "hit": False},
        ]
        self.radius_px = 35
        self.hit_count = 0
        self.total_targets = len(self.targets)

    def update(self, nx, ny, w, h, conf=1.0):
        if conf < 0.2:
            return

        gaze_x = nx * w
        gaze_y = ny * h

        for t in self.targets:
            if t["hit"]:
                continue

            tx, ty = t["pos"]
            target_x = tx * w
            target_y = ty * h

            dist_px = math.hypot(gaze_x - target_x, gaze_y - target_y)

            if dist_px < self.radius_px:
                t["hit"] = True
                self.hit_count += 1

    def draw(self, frame, w, h):
        for t in self.targets:
            tx, ty = t["pos"]
            cx = int(tx * w)
            cy = int(ty * h)
            color = (0, 255, 0) if t["hit"] else (0, 0, 255)
            thickness = -1 if t["hit"] else 2
            cv2.circle(frame, (cx, cy), self.radius_px, color, thickness)

        cv2.putText(
            frame,
            f"HITS: {self.hit_count}/{self.total_targets}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),
            2
        )
