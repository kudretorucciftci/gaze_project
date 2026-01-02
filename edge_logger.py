import csv
import time
import os

class EdgeLogger:
    def __init__(self, path="edge_log.csv", edge_th=0.035):
        self.path = path
        self.edge_th = edge_th
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "pitch",
                    "yaw",
                    "abs_pitch",
                    "abs_yaw",
                    "nx",
                    "ny",
                    "is_edge"
                ])

    def log(self, pitch, yaw, nx, ny):
        abs_p = abs(pitch)
        abs_y = abs(yaw)
        is_edge = int(abs_p > self.edge_th or abs_y > self.edge_th)

        if is_edge:
            with open(self.path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.time(),
                    pitch,
                    yaw,
                    abs_p,
                    abs_y,
                    nx,
                    ny,
                    is_edge
                ])

            print(f"[EDGE] |yaw|={abs_y:.4f}, |pitch|={abs_p:.4f}")

