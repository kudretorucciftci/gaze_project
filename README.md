# Kalibrasyonsuz Göz Takip Sistemi

Bu proje, bir web kamerası kullanarak gerçek zamanlı olarak kullanıcının göz hareketlerini takip eden ve fare imlecini ekranda buna göre hareket ettiren bir sistemdir. Herhangi bir kalibrasyon işlemi gerektirmez ve göz kırpma ile tıklama özelliğine sahiptir.

## ✨ Özellikler

- **Gerçek Zamanlı Göz Takibi:** Standart bir web kamerası ile çalışır.
- **Kalibrasyonsuz:** Kullanıcıya özel uzun kalibrasyon seansları gerektirmez.
- **Göz Kırpma ile Tıklama:** Göz kırpma hareketini algılayarak fare tıklaması yapabilir.
- **Ayarlanabilir Tıklama Hassasiyeti:** Tıklama özelliğinin hassasiyeti ve hızı kod içerisinden kolayca ayarlanabilir.
- **Akıllı İmleç Düzeltme:** Zamanla kullanıcının bakışındaki küçük sapmaları öğrenerek imleç kontrolünü iyileştirir (`implicit_bias`).

## 🚀 Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

### Gereksinimler

- Python 3.x
- `pip` (Python paket yöneticisi)

### Adımlar

1. **Projeyi Klonlayın:**
   ```sh
   git clone https://github.com/kudretorucciftci/gaze_project.git
   cd gaze_project
   ```

2. **Python Sanal Ortamı Oluşturun ve Aktive Edin:**
   Bu proje, belirli kütüphane sürümlerine ihtiyaç duymaktadır. Kütüphanelerin sistem genelindeki paketlerle çakışmaması için bir sanal ortam kullanılması şiddetle tavsiye edilir.

   ```sh
   # Windows
   python -m venv gaze_final
   .\gaze_final\Scripts\activate
   ```

3. **Bağımlılıkları Yükleyin:**
   Proje için gerekli tüm kütüphaneler `requirements.txt` dosyasında listelenmiştir. Bunları aşağıdaki komutla yükleyin:

   ```sh
   pip install -r requirements.txt
   ```

## 🏃‍♀️ Kullanım

Kurulum tamamlandıktan sonra, ana uygulamayı çalıştırmak için aşağıdaki komutu kullanın:

```sh
python ana_hat.py
```

Uygulama başladığında web kameranız açılacak ve ekranda bir pencere görünecektir. İmleciniz, göz hareketlerinizi takip etmeye başlayacaktır.

Uygulamayı kapatmak için kamera penceresi etkinken klavyeden `ESC` tuşuna basmanız yeterlidir.

## ⚙️ Yapılandırma

### Göz Kırpma ile Tıklama

Tıklama özelliğinin hassasiyetini ve hızını `ana_hat.py` dosyasının içindeki aşağıdaki değişkenleri düzenleyerek kişisel tercihinize göre ayarlayabilirsiniz:

```python
def main():
    # ...
    # --- Göz Kırpma ile Tıklama Ayarları ---
    BLINK_THRESHOLD = 0.01  # Gözün kapanma eşiği (küçük değer = daha kapalı)
    CLICK_COOLDOWN = 1.5    # Tıklamalar arası bekleme süresi (saniye)
    # ...
```

- `BLINK_THRESHOLD`: Bir göz kırpmasının algılanması için gözün ne kadar kapanması gerektiğini belirler. Değeri düşürürseniz, tıklama için gözünüzü daha belirgin kapatmanız gerekir. Değeri artırırsanız, daha hassas hale gelir.
- `CLICK_COOLDOWN`: İki tıklama arasında geçmesi gereken minimum süreyi saniye cinsinden belirler. Ardışık istenmeyen tıklamaları önlemek için kullanılır.
