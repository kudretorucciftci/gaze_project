# csv_olustur.py
import csv
import numpy as np

INPUT_CSV = "annotations.csv"
OUTPUT_CSV = "train_pitch_yaw.csv"

def screen_to_pitch_yaw(x, y):
    dx = x - 0.5
    dy = y - 0.5
    yaw = np.arctan(dx * 2.0)
    pitch = np.arctan(dy * 2.0)
    return pitch, yaw

with open(INPUT_CSV, newline="", encoding="utf-8", errors="ignore") as f_in, \
     open(OUTPUT_CSV, "w", newline="") as f_out:

    reader = csv.reader(f_in)
    writer = csv.writer(f_out)

    writer.writerow(["image_path", "pitch", "yaw"])

    for row in reader:
        # en az 3 sütun yoksa geç
        if len(row) < 3:
            continue

        path = row[0]

        # gereksiz frame'leri at
        if "_with_center" in path or "_segment" in path:
            continue

        try:
            x = float(row[1])
            y = float(row[2])
        except ValueError:
            # sayı değilse geç (header / bozuk satır)
            continue

        pitch, yaw = screen_to_pitch_yaw(x, y)
        writer.writerow([path, pitch, yaw])

print("✔ train_pitch_yaw.csv başarıyla oluşturuldu")
