# 📋 Proje Değişiklik Günlüğü (Changelog)

Bu dosya, projede yapılan önemli değişiklikleri ekibe bildirmek için tutulmaktadır.  
**Her görev tamamlandığında güncellenir.**

---

## Son Güncelleme: 2026-05-06

---

## 🔀 Git Branch (Dallanma) Sistemi ve Kullanım Rehberi

Projeyi düzenli tutmak ve ekip içi çakışmaları önlemek için aşağıdaki branch yapısı kullanılmaktadır:

### 1. Branch Yapımız
```text
main   ← Her zaman ÇALIŞAN, STABİL ve TAMAMLANMIŞ faz kodlarını içerir.
└── dev    ← (AKTİF) Tüm geliştirme, yeni özellikler ve testler burada yapılır.
```
- **`main`**: Sadece stabil kodlar burada tutulur. Doğrudan değişiklik yapılmaz.
- **`dev`**: Aktif geliştirme ortamıdır. Yeni görevler ve güncellemeler burada gerçekleştirilir.

### 2. Nasıl Çalışmalısınız?

**A. Projeyi ilk kez indiriyorsanız:**
```bash
git clone https://github.com/erigami-sl/AMR-UnderDifferentNoises-DL.git
cd AMR-UnderDifferentNoises-DL
```

**B. Aktif çalışma branch'ine geçiş yapma (ZORUNLU):**
Her zaman `dev` branch'inde çalışın:
```bash
git checkout dev
```

**C. Güncel kodları alma:**
Çalışmaya başlamadan önce her zaman güncel kodları çekin:
```bash
git pull origin dev
```

---

## 📦 Dataset Kurulumu

RML2016.10a dataset dosyası (~611MB) Git'te **yer almaz**. Aşağıdaki adımlarla erişebilirsiniz:

1. Dataset'i indirin (link ekip grubunda paylaşılmıştır)
2. **Lokal çalışma** için: `data/RML2016.10a_dict.pkl` olarak yerleştirin
3. **Google Colab** için: Google Drive'ınıza `MyDrive/AMR-Project/RML2016.10a_dict.pkl` olarak yükleyin

Detaylı talimatlar: [`data/README.md`](data/README.md)

### Doğrulama
```python
import pickle
Xd = pickle.load(open('data/RML2016.10a_dict.pkl', 'rb'), encoding='iso-8859-1')
print(len(Xd.keys()))                     # 220 olmalı
print(list(Xd.values())[0].shape)         # (1000, 2, 128) olmalı
```

---

## ✅ Tamamlanan Görevler

### Faz 1 — Baseline Reproduction

| Task | Açıklama | Durum |
|------|----------|-------|
| Task 1 | Proje yapısını yeniden düzenleme | ✅ Tamamlandı |
| Task 2 | TF1 → TF2 kod migration | ✅ Tamamlandı |
| Task 3 | Google Colab notebook oluşturma | ✅ Tamamlandı |
| Task 4 | Dataset indirme + doğrulama | ✅ Tamamlandı |
| Task 5 | Baseline model eğitimi | ✅ Tamamlandı |
| Task 6 | Baseline sonuç raporlama | ✅ Tamamlandı |

---

### Task 1: Proje Yapısını Yeniden Düzenleme
**Tarih:** 2026-05-05

#### Ne yapıldı?
- Orijinal benchmark repo'sundaki dağınık yapı kaldırıldı
- Kullanılmayan 13 model silindi, sadece **MCLDNN** ve **PET-CGDNN** tutuldu
- Tüm kod merkezi modüler yapıya taşındı

#### Proje yapısı
```
AMR-UnderDifferentNoises-DL/
├── .gitignore
├── README.md
├── CHANGELOG.md
├── requirements.txt
├── data/                   # Dataset (git'te takip EDİLMEZ)
│   └── README.md
├── notebooks/              # Google Colab notebook'ları
│   ├── 01_baseline_mcldnn.ipynb
│   └── 02_baseline_petcgdnn.ipynb
├── results/                # Eğitim sonuçları
└── src/
    ├── config.py           # Merkezi ayarlar
    ├── utils/
    │   ├── dataset.py      # Veri yükleme (MCLDNN + PET-CGDNN format desteği)
    │   └── metrics.py      # Değerlendirme araçları
    └── models/
        ├── mcldnn.py       # MCLDNN model mimarisi (TF2)
        └── petcgdnn.py     # PET-CGDNN model mimarisi (TF2)
```

---

### Task 2: TF1 → TF2 Kod Migration
**Tarih:** 2026-05-05

#### Yapılan değişiklikler

| Eski (TF1/Keras 2.2) | Yeni (TF2/tf.keras) |
|---|---|
| `from keras.layers import CuDNNLSTM` | `from tensorflow.keras.layers import LSTM` |
| `from keras.layers import CuDNNGRU` | `from tensorflow.keras.layers import GRU` |
| `import keras` / `from keras.xxx` | `from tensorflow.keras.xxx` |
| `keras.optimizers.Adam(lr=...)` | `tf.keras.optimizers.Adam(learning_rate=...)` |
| `tf.keras.backend.cos(x)` | `tf.math.cos(x)` |

> Model mimarisi (katman sayısı, filtre boyutları, aktivasyonlar) hiç değiştirilmedi.  
> Sadece TF2 API uyumluluğu sağlandı.

---

### Task 3: Google Colab Notebook Oluşturma
**Tarih:** 2026-05-05

| Notebook | Model |
|----------|-------|
| `01_baseline_mcldnn.ipynb` | MCLDNN |
| `02_baseline_petcgdnn.ipynb` | PET-CGDNN |

Her notebook: ortam kurulumu → Drive mount → repo klonlama → veri yükleme → model eğitimi → değerlendirme → sonuç kaydetme

---

### Task 4: Dataset İndirme + Doğrulama
**Tarih:** 2026-05-05

- **Dataset:** RML2016.10a — 611.2 MB
- **Modülasyonlar (11):** 8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM
- **SNR aralığı:** -20 → +18 dB (20 seviye)
- **Toplam örnek:** 220,000 (1000 × 11 mod × 20 SNR)
- **Örnek shape:** (1000, 2, 128) — 2 kanal (I/Q), 128 zaman adımı

---

### Task 5: Baseline Model Eğitimi
**Tarih:** 2026-05-05  
**Ortam:** Google Colab, T4 GPU, TensorFlow 2.x

#### PET-CGDNN Baseline Sonuçları (AWGN)

| SNR (dB) | Accuracy |
|---|---|
| -20 | 9.50% |
| -18 | 9.18% |
| -16 | 9.77% |
| -14 | 12.95% |
| -12 | 16.41% |
| -10 | 26.23% |
| -8 | 38.77% |
| -6 | 51.32% |
| -4 | 64.05% |
| -2 | 76.59% |
| 0 | 85.91% |
| +2 | 88.68% |
| +4 | 90.50% |
| +6 | 90.05% |
| +8 | 91.18% |
| **Ortalama** | **60.75%** |
| **En yüksek** | **90.68% (18 dB)** |

> MCLDNN sonuçları ayrıca eklenecek.

---

## 🔧 Uygulanan Düzeltmeler (Bug Fixes)

| Fix | Açıklama | Commit |
|-----|----------|--------|
| Modülasyon isimleri | `config.py`'deki isimler dataset ile uyumsuzdu (4-PAM→PAM4, 16-QAM→QAM16, 64-QAM→QAM64) | `2fee8b1` |
| Colab import hatası | `os.chdir()` eklenerek `ModuleNotFoundError: No module named 'src'` düzeltildi | `5380fe2` |
| Keras 3 layer isimleri | Tüm katmanlara benzersiz isim verilerek `ValueError: duplicate name` düzeltildi | `bc8872b`, `df76dd1` |
| Grafik görünmüyor | `matplotlib.use('Agg')` kaldırılıp `plt.show()` eklenerek Colab inline gösterim düzeltildi | `0f7beb5` |
| Weight format | `.h5` → `.keras` (Keras 3 önerisi) | `209746c` |

---

### Faz 2 — Kanal Modelleme ve Yeni Koşullarda Eğitim

| Task | Açıklama | Durum |
|------|----------|-------|
| Task 7 | Rayleigh ve Rician kanal modellerinin implementasyonu (`src/utils/channels.py`) | ✅ Tamamlandı |
| Task 8 | Fading etkileri uygulanmış yeni datasetlerin üretilmesi | ✅ Tamamlandı |
| Task 9 | Yeni kanal koşullarında model eğitimleri ve değerlendirmesi (Colab) | 🔜 Sırada |

---

### Task 7: Kanal Modellerinin İmplementasyonu
**Tarih:** 2026-05-05  
**Branch:** `dev`

#### Ne yapıldı?
- AWGN verisine (orijinal dataset) sönümleme etkilerini (fading) ekleyen `src/utils/channels.py` oluşturuldu.
- **Rayleigh Fading:** Sadece NLOS (görüş hattı dışı) yansımaların olduğu ortam modeli.
- **Rician Fading:** Hem LOS (doğrudan görüş) hem de NLOS yansımaların bulunduğu ortam modeli. K-faktörü ayarlanabilir.
- Model eğitimi sırasında doğrudan `(N, 2, 128)` formatındaki IQ sinyallerine uygulanacak şekilde NumPy ile vektörize olarak yazıldı (çok hızlı).
- İstenirse blok-fading (tek paket boyunca sabit sönümleme) veya fast-fading (her örnekte değişen sönümleme) seçeneği eklendi.

---

### Task 7b: Kanal Modellerinin Doğrulanması (Verification)
**Tarih:** 2026-05-05  
**Branch:** `dev`
**Notebook:** `notebooks/03_verify_channels.ipynb`

#### Ne yapıldı?
- `channels.py` implementasyonu 6 farklı test ile doğrulandı:
  1. **Shape ve hata kontrolü:** Çıkış boyutları, NaN kontrolü, geçersiz parametre hata fırlatması ✅
  2. **Rayleigh |h|² dağılımı:** Exponential(1) dağılımına uyduğu doğrulandı ✅
  3. **Rician K-faktörü:** K=0.1, 1.0, 10.0 için dağılım kontrol edildi ✅
  4. **Sinyal gücü analizi:** Fading öncesi/sonrası güç oranları ölçüldü ✅
  5. **IQ görselleştirme:** Orijinal vs faded sinyal karşılaştırması (zaman serisi + constellation) ✅
  6. **generate_faded_dataset:** Tam dataset üzerinde çalışma ve performans testi ✅

#### Tespit edilen eksikler / öneriler
| # | Bulgu | Önem | Durum |
|---|-------|------|-------|
| 1 | `seed` parametresi yok → tekrarlanabilirlik sağlanamıyor | Orta | ✅ Task 8'de eklendi |
| 2 | Fading sonrası güç normalizasyonu opsiyonu yok | Düşük | ✅ Task 8'de eklendi (`normalize_power`) |
| 3 | `__init__.py`'de channels import'u eksik | Düşük | Fonksiyonel etki yok |

---

### Task 8: Fading Uygulanmış Dataset Üretimi
**Tarih:** 2026-05-06  
**Branch:** `dev`  
**Notebook:** `notebooks/04_generate_faded_datasets.ipynb`

#### Ne yapıldı?

1. **`channels.py` iyileştirmeleri** (Task 7b bulgularının çözümü):
   - `seed` parametresi eklendi → tam tekrarlanabilirlik sağlandı
   - `normalize_power` parametresi eklendi → opsiyonel güç normalizasyonu
   - `generate_faded_dataset()` artık anahtarları sıralı işleyip deterministik sub-seed üretiyor

2. **Üretim notebook'u** (`04_generate_faded_datasets.ipynb`):
   - 3 farklı fading senaryosu için dataset üretimi
   - İstatistiksel doğrulama (güç oranları, NaN kontrolü)
   - IQ görselleştirme (zaman serisi + constellation)
   - Google Drive'a otomatik kayıt ve geri-yükleme doğrulaması

#### Üretilen dosyalar
| Dosya | Kanal | Seed | Açıklama |
|-------|-------|------|----------|
| `RML2016.10a_rayleigh.pkl` | Rayleigh | 2016 | Derin sönümleme (NLOS) |
| `RML2016.10a_rician_K3.pkl` | Rician K=3 | 3016 | Orta düzey LOS |
| `RML2016.10a_rician_K10.pkl` | Rician K=10 | 4016 | Baskın LOS |

Her dosya orijinal dataset ile aynı formatta: 220 anahtar, `(1000, 2, 128)` shape, ~611 MB.

---

### T8: Fading Etkilerinin IQ Sinyal Görselleştirmesi
**Tarih:** 2026-05-06  
**Branch:** `dev`  
**Notebook:** `notebooks/05_visualize_fading_effects.ipynb`

#### Ne yapıldı?
- 4 dataset'i (AWGN + Rayleigh + Rician K=3 + Rician K=10) yükleyerek karşılaştırmalı görsel analiz notebook'u oluşturuldu.
- **5 görselleştirme türü:**
  1. Constellation diyagramları (11 modülasyon × 4 kanal grid)
  2. I/Q zaman serisi karşılaştırması (dalga formu bozulması)
  3. SNR vs kanal etkisi matrisi (çoklu SNR seviyesi)
  4. Güç dağılımı histogramları (örnek başına güç analizi)
  5. Kanal bozulma ısı haritası (tüm mod/SNR çiftleri için güç oranı)

---

## ❓ Sıkça Sorulan Sorular

**S: Dataset neden Git'te yok?**  
C: ~611MB boyutunda, GitHub'ın 100MB dosya limiti var. Google Drive üzerinden paylaşılıyor.

**S: Kod çalışıyor mu?**  
C: Evet! Her iki model de Colab'da başarıyla eğitildi ✅.

**S: Google Colab'da nasıl çalıştıracağım?**  
C: `notebooks/` klasöründeki `.ipynb` dosyalarını Colab'a yükleyin. Runtime → GPU (T4) seçin. Hücreleri sırasıyla çalıştırın.

**S: REPO_URL ne olmalı?**  
C: `https://github.com/erigami-sl/AMR-UnderDifferentNoises-DL.git`
