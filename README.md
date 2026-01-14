# Kalibrasyonsuz Göz Takip Sistemi

Bu proje, bir web kamerası kullanarak gerçek zamanlı olarak kullanıcının göz hareketlerini takip eden ve fare imlecini ekranda buna göre hareket ettiren bir sistemdir. Herhangi bir kalibrasyon işlemi gerektirmez ve çeşitli göz hareketleriyle (göz kırpma, göz kısma, odaklanma) sol tıklama, sağ tıklama, yakınlaştırma, uzaklaştırma ve sayfa kaydırma gibi gelişmiş bilgisayar kontrolü özellikleri sunar.

## ✨ Özellikler

- **Gerçek Zamanlı Göz Takibi:** Standart bir web kamerası ile çalışır.
- **Kalibrasyonsuz:** Kullanıcıya özel uzun kalibrasyon seansları gerektirmez.
- **Akıllı İmleç Düzeltme:** Zamanla kullanıcının bakışındaki küçük sapmaları öğrenerek imleç kontrolünü iyileştirir (`implicit_bias`).
- **Gelişmiş Göz Hareketleriyle Kontrol:**
  - **Sol Tıklama:** Normal (çift gözle) kısa göz kırpma hareketi ile.
  - **Sağ Tıklama:** Sadece sağ gözü kırpma hareketi ile.
  - **Yakınlaştırma (Zoom In):** Her iki gözü kısma hareketi ile.
  - **Uzaklaştırma (Zoom Out):** Uzun göz kırpma (gözleri 1 saniyeden uzun süre kapalı tutma) ile.
  - **Akıcı Sayfa Kaydırma:** Ekranın üst veya alt kenarlarına odaklanarak.

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

2. **Eğitilmiş Model ve Veri Seti:**
   Projenin ihtiyaç duyduğu eğitilmiş model dosyası (`mpiigaze_finetuned_v2.keras`) zaten bu depoda bulunmaktadır.
   Eğer modelin ince ayar (fine-tuning) için kullanıldığı veri setini de indirmek isterseniz:
   - **Veri setini indir:** [Gaze Veri Seti](https://drive.google.com/file/d/1F-DPjKiTrWjcpQ4Pguj3wMl9axEx06x9/view?usp=drive_link)

3. **Python Sanal Ortamı Oluşturun ve Aktive Edin:**
   Bu proje, belirli kütüphane sürümlerine ihtiyaç duymaktadır. Kütüphanelerin sistem genelindeki paketlerle çakışmaması için bir sanal ortam kullanılması şiddetle tavsiye edilir.
   ```sh
   # Windows
   python -m venv gaze_final
   .\gaze_final\Scripts\activate
   ```

4. **Bağımlılıkları Yükleyin:**
   Proje için gerekli tüm kütüphaneler `requirements.txt` dosyasında listelenmiştir. Bunları aşağıdaki komutla yükleyin:
   ```sh
   pip install -r requirements.txt
   ```

## 🏃‍♀️ Kullanım

Kurulum tamamlandıktan sonra, ana uygulamayı çalıştırmak için aşağıdaki komutu kullanın:
```sh
python ana_hat.py
```
Uygulama başladığında web kameranız açılacak ve ekranda bir pencere görünecektir. İmleciniz, göz hareketlerinizi takip etmeye başlayacaktır. Uygulamayı kapatmak için kamera penceresi etkinken klavyeden `ESC` tuşuna basmanız yeterlidir.

### Göz Hareketleriyle Komutlar

- **Sol Tıklama:** İmleci istediğiniz yere getirin ve normal, **iki gözünüzle kısa bir göz kırpma** yapın.
- **Sağ Tıklama:** İmleci istediğiniz yere getirin ve sadece **sağ gözünüzü kısa bir şekilde kırpın**.
- **Yakınlaştırma (Zoom In):** İmleci yakınlaştırmak istediğiniz pencereye getirin ve **gözlerinizi hafifçe kısın**. Bu, `Ctrl + Fare Tekerleği Yukarı` komutunu taklit ederek yakınlaştırma yapar.
- **Uzaklaştırma (Zoom Out):** İmleci uzaklaştırmak istediğiniz pencereye getirin ve **gözlerinizi 1 saniyeden uzun süre kapalı tutun**. Bu, `Ctrl + Fare Tekerleği Aşağı` komutunu taklit ederek uzaklaştırma yapar.
- **Sayfa Kaydırma:** İmleci kaydırmak istediğiniz pencereye getirin.
  - **Aşağı Kaydırmak İçin:** Bakışınızı ekranın **en alt kenarına** getirin ve yaklaşık 0.6 saniye sabit tutun.
  - **Yukarı Kaydırmak İçin:** Bakışınızı ekranın **en üst kenarına** getirin ve yaklaşık 0.6 saniye sabit tutun.
  Kaydırmayı durdurmak için bakışınızı kenardan çekmeniz yeterlidir.

### Kamera Seçimi
Proje varsayılan olarak sistemdeki ilk kamerayı (genellikle 0 ID'li) kullanır. Eğer telefonunuzu (örneğin **iVCam** gibi uygulamalarla) veya başka bir harici kamerayı kullanıyorsanız, `ana_hat.py` dosyasındaki `cap = cv2.VideoCapture(0)` satırındaki `0` değerini, kullandığınız kameranın bilgisayarınızdaki cihaz ID'sine (`0`, `1`, `2` vb.) göre değiştirmeniz gerekebilir.

## 🛠️ Yöntem ve Model

Bu proje, birkaç farklı teknolojiyi bir araya getirir:

- **Yüz ve Göz Tespiti:** Yüz ve gözlerin kritik noktalarını (landmarks) gerçek zamanlı olarak tespit etmek için **Google Mediapipe** kütüphanesi kullanılmaktadır.
- **Bakış Tahmini:** Göz bölgesinden alınan görüntü, **MPIIGaze** veri seti üzerinde önceden eğitilmiş ve daha sonra kullanıcı verileriyle ince ayar (fine-tuning) yapılmış bir **TensorFlow/Keras** modeli (`mpiigaze_finetuned_v2.keras`) tarafından işlenir. Bu model, göz görüntüsünden bakışın yönünü (pitch ve yaw açıları) tahmin eder.
- **İmleç Kontrolü:** Modelden gelen tahminler, bir dizi filtreleme ve yumuşatma işleminden geçirilerek fare imlecinin akıcı bir şekilde hareket etmesi sağlanır.

## ⚙️ Yapılandırma

Tüm kontrol mekanizmalarının hassasiyetini `ana_hat.py` dosyasının içindeki aşağıdaki değişkenleri düzenleyerek kişisel tercihinize göre ayarlayabilirsiniz:

```python
def main():
    # ...
    # --- Akıcı Kaydırma Ayarları ---
    SCROLL_ZONE_HEIGHT = 70  # Ekranın üst/altındaki aktif bölge yüksekliği (piksel)
    SCROLL_ACTIVATION_DWELL = 0.6  # Kaydırmayı başlatmak için bekleme süresi (saniye)

    # --- Göz Hareketi Eylem Ayarları ---
    SQUINT_THRESHOLD = 0.019 # Göz kısma eşiği (daha büyük değer = daha hassas)
    BLINK_THRESHOLD = 0.012  # Göz kırpma eşiği (daha küçük değer = daha kapalı göz)
    ACTION_COOLDOWN = 0.8    # Eylemler arası genel bekleme süresi
    LONG_BLINK_DURATION = 0.6 # Uzun göz kırpmanın minimum süresi (saniye)
    # ...
```

- **Kaydırma Ayarları:**
    - `SCROLL_ZONE_HEIGHT`: Kaydırmayı tetikleyen kenar şeridinin kalınlığını ayarlar.
    - `SCROLL_ACTIVATION_DWELL`: Kaydırmanın başlaması için kenarda ne kadar beklemeniz gerektiğini ayarlar.
- **Eylem Ayarları:**
    - `SQUINT_THRESHOLD` ve `BLINK_THRESHOLD`: Göz kısma ve göz kırpma arasındaki hassasiyet dengesini ayarlar.
    - `ACTION_COOLDOWN`: İki komut arasında geçmesi gereken minimum süreyi belirler.
    - `LONG_BLINK_DURATION`: Bir göz kırpmasının "uzun" olarak kabul edilmesi için gereken minimum süreyi belirler.
```