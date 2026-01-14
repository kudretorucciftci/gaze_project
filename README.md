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

2. **Model Dosyasını İndirin:**
   Projenin ihtiyaç duyduğu eğitilmiş model dosyasına aşağıdaki linkten erişip indirin ve projenin ana dizinine (`.py` dosyalarıyla aynı yere) kopyalayın.
   - **Veri setini indir:** [mpiigaze_finetuned.keras](https://drive.google.com/file/d/1F-DPjKiTrWjcpQ4Pguj3wMl9axEx06x9/view?usp=drive_link)

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

### Kamera Seçimi
Proje varsayılan olarak sistemdeki ilk kamerayı (genellikle 0 ID'li) kullanır. Eğer telefonunuzu (örneğin **iVCam** gibi uygulamalarla) veya başka bir harici kamerayı kullanıyorsanız, `ana_hat.py` dosyasındaki `cap = cv2.VideoCapture(0)` satırındaki `0` değerini, kullandığınız kameranın bilgisayarınızdaki cihaz ID'sine (`0`, `1`, `2` vb.) göre değiştirmeniz gerekebilir.

Uygulama başladığında web kameranız açılacak ve ekranda bir pencere görünecektir. İmleciniz, göz hareketlerinizi takip etmeye başlayacaktır.

Uygulamayı kapatmak için kamera penceresi etkinken klavyeden `ESC` tuşuna basmanız yeterlidir.

## 🛠️ Yöntem ve Model

Bu proje, birkaç farklı teknolojiyi bir araya getirir:

- **Yüz ve Göz Tespiti:** Yüz ve gözlerin kritik noktalarını (landmarks) gerçek zamanlı olarak tespit etmek için **Google Mediapipe** kütüphanesi kullanılmaktadır.
- **Bakış Tahmini:** Göz bölgesinden alınan görüntü, **MPIIGaze** veri seti üzerinde önceden eğitilmiş ve daha sonra kullanıcı verileriyle ince ayar (fine-tuning) yapılmış bir **TensorFlow/Keras** modeli (`mpiigaze_finetuned.keras`) tarafından işlenir. Bu model, göz görüntüsünden bakışın yönünü (pitch ve yaw açıları) tahmin eder.
- **İmleç Kontrolü:** Modelden gelen tahminler, bir dizi filtreleme ve yumuşatma işleminden geçirilerek fare imlecinin akıcı bir şekilde hareket etmesi sağlanır.

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
