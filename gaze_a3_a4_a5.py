# -*- coding: utf-8 -*-
import os
import numpy as np
from collections import deque

PROFILE_PATH = "implicit_bias.npz"

# =====================================================
# A3 — TEMPORAL SMOOTHER
# =====================================================
class TemporalSmoother:
    def __init__(self, base_alpha=0.20):
        self.base_alpha = base_alpha
        self.prev_x = None
        self.prev_y = None

    def update(self, x, y):
        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y
            return x, y

        dx = x - self.prev_x
        dy = y - self.prev_y
        speed = np.linalg.norm([dx, dy])

        alpha = self.base_alpha + speed * 0.7
        alpha = np.clip(alpha, self.base_alpha, 0.35)

        self.prev_x = self.prev_x * (1 - alpha) + x * alpha
        self.prev_y = self.prev_y * (1 - alpha) + y * alpha

        return self.prev_x, self.prev_y

    def stable(self):
        return True


# =====================================================
# A4 — SPATIAL BIAS MAP
# =====================================================
class BiasMap:
    def __init__(self, grid=20):
        self.g = grid
        self.bx = np.zeros((grid, grid))
        self.by = np.zeros((grid, grid))
        self.lr = 0.002
        self.maxb = 0.12

        if os.path.exists(PROFILE_PATH):
            d = np.load(PROFILE_PATH)
            self.bx = d["bx"]
            self.by = d["by"]

    def _cell(self, x, y):
        ix = int(np.clip(x * self.g, 0, self.g - 1))
        iy = int(np.clip(y * self.g, 0, self.g - 1))
        return ix, iy

    def apply(self, x, y):
        ix, iy = self._cell(x, y)
        return (
            np.clip(x + self.bx[iy, ix], 0, 1),
            np.clip(y + self.by[iy, ix], 0, 1),
        )

    def learn(self, x, y, dx, dy, weight=1.0):
        if abs(dx) + abs(dy) < 1e-4:
            return
        ix, iy = self._cell(x, y)
        self.bx[iy, ix] = np.clip(
            self.bx[iy, ix] + dx * self.lr * weight,
            -self.maxb, self.maxb
        )
        self.by[iy, ix] = np.clip(
            self.by[iy, ix] + dy * self.lr * weight,
            -self.maxb, self.maxb
        )

    def save(self):
        np.savez(PROFILE_PATH, bx=self.bx, by=self.by)


# =====================================================
# A5 — INTENT DETECTOR
# =====================================================
class IntentDetector:
    def __init__(self, window=12, threshold=0.0005):
        self.hist = deque(maxlen=window)
        self.th = threshold

    def update(self, x, y):
        self.hist.append((x, y))
        if len(self.hist) < self.hist.maxlen:
            return False
        xs = [p[0] for p in self.hist]
        ys = [p[1] for p in self.hist]
        var = np.var(xs) + np.var(ys)
        return var < self.th
