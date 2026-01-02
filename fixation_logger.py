# -*- coding: utf-8 -*-
import time
import csv
import math

class FixationLogger:
    def __init__(self,
                 velocity_thresh=0.02,
                 min_duration=0.10,
                 refractory=0.35):

        self.velocity_thresh = velocity_thresh
        self.min_duration = min_duration
        self.refractory = refractory

        self.prev_time = None
        self.prev_point = None

        self.fix_start_time = None
        self.in_fixation = False
        self.last_fix_emit = 0.0

        self.current_fixation_time = 0.0

        self.log_file = "gaze_fixations.csv"
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "nx", "ny", "fix_event"])

    def update(self, nx, ny):
        now = time.time()

        if self.prev_point is None:
            self.prev_point = (nx, ny)
            self.prev_time = now
            return False

        dt = now - self.prev_time
        dx = nx - self.prev_point[0]
        dy = ny - self.prev_point[1]
        velocity = math.sqrt(dx*dx + dy*dy) / (dt + 1e-6)

        if velocity < self.velocity_thresh:
            if not self.in_fixation:
                self.fix_start_time = now
                self.in_fixation = True
            self.current_fixation_time = now - self.fix_start_time
        else:
            self.in_fixation = False
            self.fix_start_time = None
            self.current_fixation_time = 0.0

        fix_event = False
        if self.in_fixation:
            if self.current_fixation_time >= self.min_duration:
                if (now - self.last_fix_emit) > self.refractory:
                    fix_event = True
                    self.last_fix_emit = now

        self.prev_point = (nx, ny)
        self.prev_time = now

        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([now, nx, ny, int(fix_event)])

        return fix_event

    # 🔽 ADIM 3 İÇİN
    def fixation_duration(self):
        return self.current_fixation_time

    def is_fixating(self):
        return self.in_fixation
