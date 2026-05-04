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
| `src/models/mcldnn.py` | MCLDNN model tanımı (henüz TF1 importları — Task 2'de TF2'ye migrate edilecek) |
| `src/models/petcgdnn.py` | PET-CGDNN model tanımı (henüz TF1 importları — Task 2'de TF2'ye migrate edilecek) |

---

## ⏳ Sıradaki Görevler

| Task | Açıklama | Durum |
|------|----------|-------|
| Task 2 | TF1 → TF2 kod migration | 🔜 Sırada |
| Task 3 | Google Colab notebook oluşturma | ⏳ Beklemede |
| Task 4 | Dataset indirme + doğrulama (Colab) | ⏳ Beklemede |
| Task 5 | Baseline model eğitimi (Colab GPU) | ⏳ Beklemede |
| Task 6 | Baseline sonuç raporlama | ⏳ Beklemede |

---

## ❓ Sıkça Sorulan Sorular

**S: Dataset neden Git'te yok?**  
C: ~500MB boyutunda, GitHub'ın 100MB dosya limiti var. Google Drive üzerinden paylaşılıyor.

**S: Hangi branch üzerinde çalışmalıyım?**  
C: `dev/project-restructure`. Main branch'e dokunmayın.

**S: Kod henüz çalışıyor mu?**  
C: Hayır. Model dosyaları (`src/models/`) hâlâ TF1 importları içeriyor. Task 2'den sonra TF2 ile çalışır hale gelecek.

**S: Google Colab'da nasıl çalıştıracağım?**  
C: Task 3'te Colab notebook'ları oluşturulacak. O zamana kadar bekleyin.
