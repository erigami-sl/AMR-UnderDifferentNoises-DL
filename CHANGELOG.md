# 📋 Proje Değişiklik Günlüğü (Changelog)

Bu dosya, projede yapılan önemli değişiklikleri ekibe bildirmek için tutulmaktadır.  
**Her görev tamamlandığında güncellenir.**

---

## Son Güncelleme: 2026-05-05

---

## 🔀 Git Kullanım Rehberi

### Repo'yu klonlama (ilk kez)
```bash
git clone <REPO_URL>
cd AMR-UnderDifferentNoises-DL
```

### Geliştirme branch'ine geçiş
Tüm aktif geliştirme `dev/project-restructure` branch'inde yapılmaktadır.  
`main` branch'i dokunulmamaktadır.

```bash
git checkout dev/project-restructure
```

### Güncellemeleri alma
```bash
git pull origin dev/project-restructure
```

### Branch yapısı
```
main                        ← Stabil, dokunulmaz
└── dev/project-restructure ← Aktif geliştirme branch'i (BURAYA GEÇİN)
```

---

## 📦 Dataset Kurulumu

RML2016.10a dataset dosyası (~500MB) Git'te **yer almaz**. Aşağıdaki adımlarla erişebilirsiniz:

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

### Task 1: Proje Yapısını Yeniden Düzenleme
**Tarih:** 2026-05-05  
**Branch:** `dev/project-restructure`  
**Commit'ler:**
- `c1230e9` — restructure project into modular src/ layout
- `b952e2e` — remove legacy RML201610a/, add dataset setup guide

#### Ne yapıldı?
- Orijinal benchmark repo'sundaki **dağınık yapı** (her model için ayrı dosya kopyaları) kaldırıldı
- Kullanılmayan 13 model silindi, sadece **MCLDNN** ve **PET-CGDNN** tutuldu
- Tüm kod **merkezi modüler yapıya** taşındı

#### Yeni proje yapısı
```
AMR-UnderDifferentNoises-DL/
├── .gitignore              # Git dışı bırakılan dosyalar
├── README.md               # Proje açıklaması
├── CHANGELOG.md            # Bu dosya (ekip bilgilendirme)
├── requirements.txt        # Python bağımlılıkları
│
├── data/                   # Dataset dizini (git'te takip EDİLMEZ)
│   └── README.md           # Dataset indirme talimatları
│
├── notebooks/              # Google Colab notebook'ları
│
├── results/                # Eğitim sonuçları (weight, figure, prediction)
│
└── src/                    # KAYNAK KOD
    ├── config.py           # Merkezi ayarlar (path, hyperparameter, ortam algılama)
    ├── utils/
    │   ├── dataset.py      # Veri yükleme (her iki model için tek dosya)
    │   └── metrics.py      # Değerlendirme araçları (confusion matrix, SNR grafikleri)
    └── models/
        ├── mcldnn.py       # MCLDNN model mimarisi
        └── petcgdnn.py     # PET-CGDNN model mimarisi
```

#### Önemli dosyalar ve ne işe yararlar

| Dosya | Açıklama |
|-------|----------|
| `src/config.py` | Tüm path'ler, hyperparametreler ve Colab/lokal ortam algılama burada. Hardcoded path **YOK**. |
| `src/utils/dataset.py` | Veri yükleme + her model için farklı input formatı hazırlama (`prepare_data_mcldnn`, `prepare_data_petcgdnn`) |
| `src/utils/metrics.py` | Confusion matrix, SNR vs accuracy grafikleri, tam değerlendirme pipeline'ı (`evaluate_model`) |
| `src/models/mcldnn.py` | MCLDNN model tanımı (**TF2 uyumlu** ✅) |
| `src/models/petcgdnn.py` | PET-CGDNN model tanımı (**TF2 uyumlu** ✅) |

---

## ✅ Task 2: TF1 → TF2 Kod Migration
**Tarih:** 2026-05-05  
**Commit:** `[migration] TF1→TF2: migrate MCLDNN and PET-CGDNN to tf.keras`

#### Ne yapıldı?
Her iki model dosyası TensorFlow 2 / tf.keras ile uyumlu hale getirildi.

#### Yapılan değişiklikler

| Eski (TF1/Keras 2.2) | Yeni (TF2/tf.keras) | Neden? |
|---|---|---|
| `from keras.layers import CuDNNLSTM` | `from tensorflow.keras.layers import LSTM` | CuDNNLSTM TF2'de kaldırıldı. TF2 LSTM, GPU varsa otomatik CuDNN kullanır |
| `from keras.layers import CuDNNGRU` | `from tensorflow.keras.layers import GRU` | Aynı sebep |
| `import keras` / `from keras.xxx` | `from tensorflow.keras.xxx` | TF2'de keras, tf.keras altında birleşti |
| `keras.optimizers.Adam(lr=...)` | `tf.keras.optimizers.Adam(learning_rate=...)` | `lr` parametresi deprecated |
| `tf.keras.backend.cos(x)` | `tf.math.cos(x)` | Daha doğrudan API |
| `from keras.layers.convolutional import Conv2D` | `from tensorflow.keras.layers import Conv2D` | Alt modül yapısı değişti |
| `from keras.utils.vis_utils import plot_model` | Kaldırıldı (gerektiğinde `tf.keras.utils.plot_model`) | Gereksiz import |

#### Dosyalara eklenen dokümantasyon
- Her iki model dosyasına paper referansı, mimari açıklaması ve parametre dokümantasyonu eklendi
- `__main__` bloğu TF2 uyumlu hale getirildi

> **Not:** Model mimarisi (katman sayısı, filtre boyutları, aktivasyonlar) hiç değiştirilmedi.  
> Sadece TF2 API uyumluluğu sağlandı.

## ✅ Task 3: Google Colab Notebook Oluşturma
**Tarih:** 2026-05-05  
**Commit:** `[notebooks] add baseline training notebooks for MCLDNN and PET-CGDNN`

#### Oluşturulan notebook'lar

| Notebook | Model | Konum |
|----------|-------|-------|
| `01_baseline_mcldnn.ipynb` | MCLDNN | `notebooks/` |
| `02_baseline_petcgdnn.ipynb` | PET-CGDNN | `notebooks/` |

#### Her notebook şunları içerir:
1. **Ortam kurulumu** — GPU kontrolü, TF versiyon doğrulama
2. **Drive mount + Repo klonlama** — `git clone -b dev/project-restructure`
3. **Dataset yükleme** — Drive veya lokal path otomatik algılama
4. **Veri görselleştirme** — Örnek IQ sinyalleri (6 farklı modülasyon)
5. **Model oluşturma** — `src/models/` modüllerinden import
6. **Eğitim** — EarlyStopping, ReduceLR, ModelCheckpoint
7. **Değerlendirme** — Confusion matrix, SNR vs accuracy, per-mod accuracy
8. **Sonuç kaydetme** — Ağırlıklar + sonuçlar Drive'a kaydedilir

#### ⚠️ Kullanım öncesi yapılması gerekenler:
1. Notebook'ta `REPO_URL` değişkenini kendi repo URL'niz ile güncelleyin
2. Dataset'i Drive'a yükleyin: `MyDrive/AMR-Project/RML2016.10a_dict.pkl`
3. Colab'da Runtime → Change runtime type → **GPU (T4)** seçin

---

## ⏳ Sıradaki Görevler

| Task | Açıklama | Durum |
|------|----------|-------|
| Task 1 | Proje yapısını yeniden düzenleme | ✅ Tamamlandı |
| Task 2 | TF1 → TF2 kod migration | ✅ Tamamlandı |
| Task 3 | Google Colab notebook oluşturma | ✅ Tamamlandı |
| Task 4 | Dataset indirme + doğrulama (Colab) | 🔜 Sırada |
| Task 5 | Baseline model eğitimi (Colab GPU) | ⏳ Beklemede |
| Task 6 | Baseline sonuç raporlama | ⏳ Beklemede |

---

## ❓ Sıkça Sorulan Sorular

**S: Dataset neden Git'te yok?**  
C: ~500MB boyutunda, GitHub'ın 100MB dosya limiti var. Google Drive üzerinden paylaşılıyor.

**S: Hangi branch üzerinde çalışmalıyım?**  
C: `dev/project-restructure`. Main branch'e dokunmayın.

**S: Kod henüz çalışıyor mu?**  
C: Evet! Notebook'lar hazır ✅. Dataset'i Drive'a yükleyip Colab'da çalıştırabilirsiniz.

**S: Google Colab'da nasıl çalıştıracağım?**  
C: `notebooks/` klasöründeki `.ipynb` dosyalarını Colab'a yükleyin. İlk hücrede repo klonlanır, Drive mount edilir ve eğitim başlar.

**S: REPO_URL ne olmalı?**  
C: GitHub'daki repo URL'niz. Örnek: `https://github.com/KULLANICI_ADI/AMR-UnderDifferentNoises-DL.git`

