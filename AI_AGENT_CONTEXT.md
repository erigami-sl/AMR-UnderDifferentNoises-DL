# 🤖 AI Agent Project Context & Master Guide

This file is the single source of truth for the project. It replaces the original `Project Proposal.odt`, merges the technical `project_analysis`, and serves as a comprehensive manual for any AI agent (GitHub Copilot, Cursor, etc.) working on this repository.

## 1. AI Agent Operating Rules (CRITICAL)
- **Rule 1: Always update CHANGELOG.md:** After completing any task, you MUST log your actions, changes, and bug fixes in `CHANGELOG.md` so the human team knows exactly what was done.
- **Rule 2: Commit after every task:** You MUST `git add` and `git commit` your changes logically after finishing a distinct piece of work. Include descriptive commit messages and push to the remote branch frequently.
- **Rule 3: Branching Discipline:** Never develop on `main`. All work must be done on the `dev` branch. `main` is strictly for fully tested, completed phase releases.

## 2. Project Proposal & Background
- **Problem Statement:** Automatic Modulation Recognition (AMR) plays a crucial role in signal processing, spectrum monitoring, and cognitive radio. Manual feature extraction is tedious; ML models automate this to reduce time and improve efficiency.
- **Specific Challenges:** Noise and channel characteristics are non-deterministic. Signal-to-Noise Ratio (SNR) heavily impacts accuracy.
- **Objective:** Evaluate and improve AMR under different SISO channel conditions (AWGN, Rayleigh, Rician) using Deep Learning. Focus on low-SNR environments to make the models robust and suitable for practical, commercial use.
- **References:**
  - *Main:* Zhang, F., Luo, C., et al. (2022). Deep learning based automatic modulation recognition: Models, datasets, and challenges. Digital Signal Processing.
  - *Additional:* O'Shea, J. et al. (2018). Over-the-Air Deep Learning Based Radio Signal Classification. IEEE.
- **Baseline Repositories:** [AMR-Benchmark](https://github.com/Richardzhangxx/AMR-Benchmark) and [CommPy](https://github.com/veeresht/CommPy).

## 3. Dataset Description
- **Dataset:** RadioML 2016.10a (DeepSig, GNU Radio).
- **Size:** 220,000 samples across 11 modulation classes (8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM).
- **Format:** `(1000, 2, 128)` tensors representing I/Q channels. (1000 samples per modulation/SNR pair).
- **SNR Range:** -20 to +18 dB (2 dB steps).
- **Preprocessing:** 60/20/20 train/val/test split. Signal normalization applied.
- **Location:** Ignored in git due to size (~611MB). Must be placed in `data/RML2016.10a_dict.pkl` or loaded from Google Drive in Colab (`MyDrive/AMR-Project/RML2016.10a_dict.pkl`).

## 4. Codebase Architecture & Technical Details
The codebase was restructured from a scattered benchmark repository into a modular Python package:
```text
AMR-UnderDifferentNoises-DL/
├── data/                   # (Ignored) Place RML2016.10a_dict.pkl here
├── notebooks/              # Google Colab notebooks for training and dataset generation
├── results/                # Evaluation results, weights, and plots
└── src/
    ├── config.py           # Global settings, class names, path resolution
    ├── models/
    │   ├── mcldnn.py       # MCLDNN architecture
    │   └── petcgdnn.py     # PET-CGDNN architecture
    └── utils/
        ├── channels.py     # Rayleigh & Rician fading simulation pipeline
        ├── dataset.py      # Data loading and preprocessing pipelines
        └── metrics.py      # Confusion matrix, accuracy plots, evaluation
```

### Technical Migration Notes (TF1 to TF2/Keras 3)
- Original repo was Python 3.6 + TF 1.14. We use TF 2.15+ (Keras 3).
- `CuDNNLSTM`/`CuDNNGRU` were replaced with standard `LSTM`/`GRU` (which auto-use CuDNN on GPU).
- **CRITICAL Keras 3 Requirement:** All Keras layers must have explicit, unique `name` parameters. Auto-naming causes `ValueError: duplicate name` in Keras 3.
- Colab plotting requires standard `plt.show()`. Do not use `matplotlib.use('Agg')` as it breaks inline outputs in notebooks.

### Channel Modeling (CommPy & Numpy)
- Implemented in `src/utils/channels.py`.
- Complex baseband representation: `I + jQ`. Fading applied via `h * (I + jQ)`.
- **Rayleigh Fading:** Fading coefficient `h ~ CN(0, 1)`. Simulates Non-Line-of-Sight (NLOS) environments.
- **Rician Fading:** Fading coefficient has a Line-of-Sight (LOS) component determined by the K-factor, plus an NLOS component.
- Implemented using highly optimized NumPy vectorization over `(N, 2, 128)` batches.

## 5. Project Phases and Work Plan
### Phase 1: Environment & Baseline (Week 5-7) — [COMPLETED]
- Dataset downloaded, preprocessed, and formatted.
- Code migrated from TF1/Keras2 to TF2/Keras3.
- Baseline models (MCLDNN and PET-CGDNN) reproduced from original implementations on the AWGN channel.
- Colab baseline evaluation pipelines established.

### Phase 2: Channel Modeling & Generation (Week 9) — [ACTIVE]
- Integrate Rayleigh and Rician fading channels.
- Generate noisy datasets synthetically across multiple SNR levels using `src/utils/channels.py`.
- Visualize the fading effects on IQ signals.
- Complete initial training runs on these new fading datasets.

### Phase 3: Full Evaluation & Tuning (Week 12)
- Collect full experimental results across all channel conditions.
- Complete hyperparameter tuning (learning rate, batch size, patience).
- Finalize evaluation metrics (accuracy, confusion matrices, SNR vs. accuracy curves).
- Analyze known confusion pairs (e.g., 16QAM/64QAM and WBFM/AM-DSB) under low-SNR.

### Phase 4: Reporting & Finalization (Week 13-14)
- Presentation delivery.
- Final report and documented code submission.
