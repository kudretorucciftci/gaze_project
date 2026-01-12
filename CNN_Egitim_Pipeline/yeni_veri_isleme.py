# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
from scipy.io import loadmat
from multiprocessing import Pool, cpu_count
import tqdm
import cv2

# --- Ayarlar ---
DATA_DIR = "MPIIGaze/Data/Normalized"
OUTPUT_FILE = "mpiigaze_processed.npz"
IMAGE_DIMS = (36, 60) # (yükseklik, genişlik)

def process_mat_file(file_path):
    """
    Tek bir .mat dosyasını işler, içindeki tüm örnekleri (görüntü, bakış) çıkarır.
    """
    try:
        mat = loadmat(file_path, squeeze_me=True, struct_as_record=False)
        data = mat['data']
        
        results = []
        
        # --- Sol Göz ---
        left_images = data.left.image
        left_gazes = data.left.gaze
        for i in range(len(left_images)):
            img = left_images[i].astype(np.float32) / 255.0
            gaze = left_gazes[i]
            
            img = cv2.flip(img, 1)
            
            pitch = np.arcsin(-gaze[1])
            yaw = np.arctan2(-gaze[0], -gaze[2])
            yaw = -yaw
            
            results.append((img, np.array([pitch, yaw])))

        # --- Sağ Göz ---
        right_images = data.right.image
        right_gazes = data.right.gaze
        for i in range(len(right_images)):
            img = right_images[i].astype(np.float32) / 255.0
            gaze = right_gazes[i]

            pitch = np.arcsin(-gaze[1])
            yaw = np.arctan2(-gaze[0], -gaze[2])
            
            results.append((img, np.array([pitch, yaw])))
            
        return results
        
    except Exception as e:
        # print(f"Hata: {file_path} işlenirken sorun oluştu: {e}")
        return []

def main():
    print("MPIIGaze veri seti işleniyor (Doğru Metot)...")
    
    file_list = []
    for i in range(15):
        participant = f"p{i:02d}"
        path_pattern = os.path.join(DATA_DIR, participant, "day*.mat")
        file_list.extend(glob.glob(path_pattern))
        
    if not file_list:
        print(f"HATA: '{DATA_DIR}' içinde hiç .mat dosyası bulunamadı.")
        return

    print(f"Toplam {len(file_list)} adet dosya bulundu.")
    
    num_processes = cpu_count()
    print(f"{num_processes} çekirdek ile paralel işleme başlatılıyor...")
    
    all_images = []
    all_gazes = []
    
    with Pool(processes=num_processes) as pool:
        for result in tqdm.tqdm(pool.imap_unordered(process_mat_file, file_list), total=len(file_list), desc="Dosyalar işleniyor"):
            for img, gaze in result:
                # Görüntülerin şekli zaten tutarlı olmalı (36, 60)
                all_images.append(img[..., np.newaxis])
                all_gazes.append(gaze)

    if not all_images:
        print("HATA: Hiçbir dosya başarıyla işlenemedi.")
        return

    X_data = np.array(all_images, dtype=np.float32)
    y_data = np.array(all_gazes, dtype=np.float32)

    print(f"\nİşleme tamamlandı. Toplam {len(X_data)} adet örnek çıkarıldı.")
    print(f"X_data şekli: {X_data.shape}")
    print(f"y_data şekli: {y_data.shape}")
    
    print(f"Veriler '{OUTPUT_FILE}' dosyasına kaydediliyor...")
    np.savez_compressed(OUTPUT_FILE, x=X_data, y=y_data)
    
    print("✅ Başarıyla tamamlandı!")

if __name__ == '__main__':
    main()