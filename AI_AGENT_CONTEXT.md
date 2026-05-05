# 🤖 AI Agent Project Context

This file is intended for other AI agents (like GitHub Copilot, Cursor, or other coding assistants) joining the project. It provides a comprehensive summary of the project proposal, architecture, current state, and git rules.

## 1. Project Background and Proposal
- **Problem Statement:** In communication networks, Automatic Modulation Recognition (AMR) plays a crucial role in signal processing, spectrum monitoring, and cognitive radio. Traditional AMR requires extensive human expertise and manual feature extraction. Machine Learning (ML) models make this selection process automatic, reducing time and improving efficiency.
- **Specific Challenges:** Noise and channel characteristics are non-deterministic and challenging to model. In communication, Signal-to-Noise Ratio (SNR) is a major issue that can lead to incorrect outcomes.
- **Objective:** Evaluate and improve AMR under different SISO channel conditions (AWGN, Rayleigh, Rician) using Deep Learning, specifically aiming to develop models suitable for low-SNR systems for commercial use.
- **Academic References:** 
  - Main: *Zhang, F., Luo, C., ... Deep learning based automatic modulation recognition: Models, datasets, and challenges. Digital Signal Processing, 2022.*
  - Additional: *O'Shea, J., ... Over-the-Air Deep Learning Based Radio Signal Classification. IEEE, 2018.*
- **Baseline Repositories:** [AMR-Benchmark](https://github.com/Richardzhangxx/AMR-Benchmark) and [CommPy](https://github.com/veeresht/CommPy).

## 2. Dataset Description
- **Source:** RadioML 2016.10a (DeepSig, GNU Radio).
- **Size:** 220,000 samples across 11 modulation classes (8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM).
- **Format:** `(1000, 2, 128)` tensors representing I/Q channels.
- **SNR Range:** -20 to +18 dB.
- **Location:** Not in git (too large). Must be placed in `data/RML2016.10a_dict.pkl` or loaded from Google Drive in Colab.

## 3. Project Phases and Timeline
- **Phase 1: Environment & Baseline (Week 5-7) — [COMPLETED]**
  - Dataset downloaded, preprocessed, and formatted (60/20/20 split).
  - Code migrated from TF1/Keras2 to TF2/Keras3.
  - Baseline models (MCLDNN and PET-CGDNN) reproduced from original implementations on the AWGN channel.
- **Phase 2: Channel Modeling & Generation (Week 9) — [ACTIVE]**
  - Integrate Rayleigh and Rician fading channels using `src/utils/channels.py`.
  - Generate noisy datasets synthetically across multiple SNR levels.
  - Complete initial training runs on these new datasets.
- **Phase 3: Full Evaluation & Tuning (Week 12)**
  - Collect full experimental results across all channel conditions.
  - Complete hyperparameter tuning.
  - Finalize evaluation metrics (accuracy, confusion matrices, SNR vs. accuracy curves).
  - Analyze known confusion pairs (16QAM/64QAM and WBFM/AM-DSB) under low-SNR.
- **Phase 4: Reporting & Finalization (Week 13-14)**
  - Presentation delivery.
  - Final report and documented code submission.

## 4. Codebase Architecture
The codebase was restructured from a scattered benchmark repository into a modular Python package:
```text
AMR-UnderDifferentNoises-DL/
├── data/                   # (Ignored in git) Place RML2016.10a_dict.pkl here
├── notebooks/              # Google Colab notebooks for training and dataset generation
├── results/                # Evaluation results, weights, and plots
└── src/
    ├── config.py           # Global settings, class names, path resolution
    ├── models/
    │   ├── mcldnn.py       # Keras 3 compatible MCLDNN architecture
    │   └── petcgdnn.py     # Keras 3 compatible PET-CGDNN architecture
    └── utils/
        ├── channels.py     # Rayleigh & Rician fading simulation pipeline
        ├── dataset.py      # Data loading and preprocessing pipelines
        └── metrics.py      # Confusion matrix, accuracy plots, evaluation
```

## 5. Git Strategy & Branching Rules
- **`main` branch:** STABLE ONLY. Do not push incomplete or untested code here.
- **`dev` branch:** ACTIVE DEVELOPMENT. All new features, tests, and task implementations must be done and pushed here.

## 6. Key Implementation Details
- **Keras 3 Compatibility:** All Keras layers must have explicit, unique `name` parameters to avoid `ValueError: duplicate name` when using Keras 3.
- **Plotting in Colab:** `matplotlib.use('Agg')` must NOT be used if inline plotting is required in Colab notebooks. We use standard `plt.show()`.
- **Channel Modeling (`src/utils/channels.py`):** Fading is implemented via complex multiplication `(I + jQ) * h`. `h` is generated using numpy for vectorized speed over batches of `(N, 2, 128)`.
