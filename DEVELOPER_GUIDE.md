# 📚 Ekip İçin Kapsamlı Geliştirme Rehberi (Developer Guide)

Hoş geldiniz! Bu rehber, AMR (Automatic Modulation Recognition) projemizde ekip üyelerinin nasıl kod yazacağı, test yapacağı ve projeyi nasıl büyüteceği konusunda adım adım ve çok kapsamlı bilgiler içerir.

Lütfen kod yazmaya başlamadan önce bu dokümanı dikkatlice okuyun.

---

## 1. Projenin Amacı ve Temel Kavramlar

Bu projenin amacı, **düşük SNR (Sinyal/Gürültü Oranı)** ve **zorlu kanal koşullarında (Rayleigh, Rician fading)** çalışan derin öğrenme modellerinin başarımını test etmek ve iyileştirmektir. 
- **Dataset:** `RadioML 2016.10a` (11 Modülasyon Sınıfı, -20 dB ile +18 dB arası 20 farklı SNR seviyesi).
- **Giriş Formatı:** Sinyaller zaman ekseninde I ve Q kanalları olmak üzere `(2, 128)` formatında verilir. Projemizde veri işleme sırasında batch boyutu ile birlikte `(N, 2, 128)` formatında çalışılır.
- **Modeller:** Şu an `MCLDNN` ve `PET-CGDNN` olmak üzere 2 ana model desteklenmektedir. Yeni modeller eklenebilir.

---

## 2. Ortam Kurulumu (Environment Setup)

Projeyi iki farklı ortamda çalıştırabilirsiniz: Kendi bilgisayarınızda (Lokal) veya Google Colab'da (Önerilen).

### A. Lokal Kurulum (Kendi Bilgisayarınız)
Kodu test etmek, yeni modeller yazmak ve debug yapmak için lokal kurulum önerilir.
```bash
# 1. Projeyi klonlayın ve dev branch'ine geçin
git clone https://github.com/erigami-sl/AMR-UnderDifferentNoises-DL.git
cd AMR-UnderDifferentNoises-DL
git checkout dev

# 2. Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt
```

### B. Google Colab (Ağır Eğitimler İçin)
Eğitimler (Training) ekran kartı (GPU) gerektirir. Colab'da çalışırken GPU'yu açmayı unutmayın (`Runtime > Change runtime type > T4 GPU`).
Notebook'lar otomatik olarak GitHub'dan `dev` branch'ini klonlayacak şekilde ayarlanmıştır.

---

## 3. Dataset Kurulumu

Dataset (~611MB) Git'te barındırılmaz.
1. **Veriyi indirin:** Ekip içi paylaşılan linkten `RML2016.10a_dict.pkl` dosyasını indirin.
2. **Lokalde çalışıyorsanız:** Dosyayı projenin içindeki `data/` klasörüne kopyalayın (`data/RML2016.10a_dict.pkl` olacak şekilde).
3. **Colab'da çalışıyorsanız:** Dosyayı Google Drive'ınızda tam olarak şu yola koyun: `MyDrive/AMR-Project/RML2016.10a_dict.pkl`. (Notebook'lar bu yolu otomatik tanır).

---

## 4. Git ve Branch (Dallanma) Stratejisi

Karmaşayı önlemek için çok basit bir kuralımız var:
- **`main`**: Yalnızca tamamlanmış, test edilmiş fazlar burada durur. Asla `main`'e doğrudan kod yollamayın.
- **`dev`**: Çalışma alanımızdır. Tüm kodlar buraya yazılır.

**Günlük Çalışma Akışı:**
```bash
# 1. Her sabah veya çalışmaya başlamadan önce güncel kodu çekin:
git pull origin dev

# 2. Kodunuzu yazın, modelleri test edin...

# 3. Yaptığınız işi CHANGELOG.md dosyasına mutlaka ekleyin!

# 4. İşiniz bitince kaydedin ve push'layın:
git add .
git commit -m "[özellik/bug_fix] Yaptığınız işin kısa açıklaması"
git push origin dev
```

---

## 5. Proje Mimarisi Nasıl Çalışır?

Projemiz modüler bir yapıdadır. Kod yazacağınız zaman dosyaların yerleri şu şekildedir:

*   **`src/config.py`**: Projedeki tüm global ayarlar (klasör yolları, modülasyon isimleri, epoch sayıları). Bir ayar değiştirecekseniz buraya bakın.
*   **`src/utils/dataset.py`**: Pickle dosyasını okuyan, 60/20/20 train/val/test olarak bölen ve modellere özel (reshape/swapaxes) işlemleri yapan kısımdır.
*   **`src/utils/metrics.py`**: Karışıklık matrisi (Confusion Matrix) ve SNR-Accuracy grafikleri çizen fonksiyonlar buradadır.
*   **`src/utils/channels.py`**: Gürültü (AWGN) dışındaki asıl kanal modellerimiz (Rayleigh, Rician fading) burada simüle edilir.
*   **`src/models/`**: Tüm derin öğrenme modellerimiz (`mcldnn.py`, `petcgdnn.py` vb.) burada yer alır.

---

## 6. Adım Adım: Yeni Bir Model Eklemek

Eğer projeye yeni bir model (Örn: ResNet veya kendi tasarladığınız bir CNN) eklemek isterseniz, şu adımları izleyin:

### Adım 1: Model dosyasını oluşturun
`src/models/` içine yeni bir dosya açın (Örn: `yeni_model.py`).

### Adım 2: Keras 3 uyumlu şekilde mimariyi yazın
**DİKKAT:** Projemiz TensorFlow 2 ve Keras 3 kullanmaktadır. Keras 3'ün en büyük özelliği, tüm katmanların (layers) eşsiz bir isme (`name=` parametresi) sahip olmasını zorunlu kılmasıdır. Aksi takdirde `ValueError` alırsınız.

**Doğru bir model yazım örneği:**
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten

def YeniModel(weights=None, input_shape=(2, 128), classes=11, **kwargs):
    # Girdi katmanı
    inp = Input(shape=input_shape, name='input_yeni_model')
    
    # Tüm katmanlara mutlaka eşsiz bir 'name' verin!
    x = Conv2D(64, (1, 3), activation='relu', name='conv1')(inp)
    x = Flatten(name='flatten_katmani')(x)
    
    # Çıktı katmanı (softmax)
    out = Dense(classes, activation='softmax', name='softmax_ciktisi')(x)
    
    model = Model(inputs=inp, outputs=out)
    return model
```

### Adım 3: Dataset loader'a format ekleyin
Her model I/Q sinyallerini farklı formatta isteyebilir. Eğer modeliniz `(128, 2)` istiyorsa, `src/utils/dataset.py` içine formatlayıcı bir fonksiyon yazın (Örn: `prepare_data_yenimodel`).

### Adım 4: Colab Notebook oluşturun
`notebooks/` klasörüne `04_yenimodel_training.ipynb` adında bir dosya kopyalayın. `src.models.yeni_model` üzerinden modelinizi import edip eğitime başlayın.

---

## 7. Fading (Kanal Bozulumu) Uygulamak

Faz 2 kapsamında projeye Rayleigh ve Rician sönümlemeleri eklendi. Bu fonksiyonları şu şekilde kullanabilirsiniz:

```python
from src.utils.channels import apply_fading

# X_train formatı: (N, 2, 128) olmalıdır.
# Rayleigh kanalı uygulama (Sadece NLOS yansımalar)
X_train_rayleigh = apply_fading(X_train, channel_type='rayleigh')

# Rician kanalı uygulama (K-faktörü ayarlanabilir, LOS + NLOS)
X_train_rician = apply_fading(X_train, channel_type='rician', K=5.0)
```
Bu fonksiyon sinyalleri doğrudan işler ve orijinal noise'un (AWGN) üzerine kanal zayıflaması ekler. Modelleri bu üretilen verilerle eğitecek veya test edeceksiniz.

---

## 8. Altın Kurallar (Ekip İçi Disiplin)

1.  **Hardcoded Path Kullanmayın:** Klasör yollarını elinizle yazmayın (`C:/Ahmet/Dataset/` gibi). Her zaman `src.config` içindeki `DATASET_PATH` veya `PROJECT_DIR` gibi değişkenleri kullanın. Kodunuz başkasının bilgisayarında da çalışmalı!
2.  **CHANGELOG'u Güncelleyin:** Eğer repoya yeni bir model eklerseniz veya bir bug çözerseniz, push atmadan hemen önce `CHANGELOG.md` dosyasını açıp ne yaptığınızı yazın. Aksi takdirde diğerleri kodunuzdaki değişimi anlayamaz.
3.  **H5 yerine Keras Formatı:** Keras 3 ile birlikte `.h5` uzantısı legacy (eski) kabul ediliyor. Model ağırlıklarını kaydederken mutlaka `.keras` uzantısı kullanın (`model.save('weights.keras')`).
4.  **Colab'da Çizimler (Plotting):** `metrics.py` içindeki grafik kodlarında `plt.show()` kullandık. Eğer bir grafiğin notebook'ta görünmediğini fark ederseniz kodun sonunda `plt.show()` olup olmadığını kontrol edin.

Kolay gelsin! Projeye katkılarınız için teşekkürler. 🚀
