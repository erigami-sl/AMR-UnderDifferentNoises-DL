# 🤖 AI Agent Project Context

This file is intended for other AI agents (like GitHub Copilot, Cursor, or other coding assistants) joining the project. It provides a quick summary of the project architecture, current state, and rules.

## 1. Project Overview
- **Objective:** Evaluate and improve Automatic Modulation Recognition (AMR) under different SISO channel conditions (AWGN, Rayleigh, Rician) using Deep Learning.
- **Dataset:** RadioML 2016.10a (11 modulations, 20 SNR levels from -20 to +18 dB). The dataset is in `(1000, 2, 128)` format (I/Q channels). 
- **Dataset Location:** Not in git (too large). Must be placed in `data/RML2016.10a_dict.pkl` or loaded from Google Drive in Colab.
- **Models:** MCLDNN and PET-CGDNN (migrated from TF1/Keras2 to TF2/tf.keras).

## 2. Codebase Architecture
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
        ├── channels.py     # Rayleigh & Rician fading simulation (Phase 2)
        ├── dataset.py      # Data loading and preprocessing pipelines
        └── metrics.py      # Confusion matrix, accuracy plots, evaluation
```

## 3. Current Phase and Git Strategy
- **Phase 1 (Completed):** Baseline reproduction on AWGN channel. Code migrated to TF2, models trained, and evaluation pipelines created.
- **Phase 2 (Active):** Channel modeling. Simulating Rayleigh and Rician fading on the AWGN dataset to evaluate model degradation.

### Git Rules for AI Agents:
- **`main` branch:** STABLE ONLY. Do not push incomplete or untested code here.
- **`dev` branch:** ACTIVE DEVELOPMENT. All new features, tests, and task implementations should be done and pushed here.

## 4. Key Implementation Details
- **Keras 3 Compatibility:** All Keras layers must have explicit, unique `name` parameters to avoid `ValueError: duplicate name` when using Keras 3 (especially in loops or shared layers).
- **Plotting in Colab:** `matplotlib.use('Agg')` must NOT be used if inline plotting is required in Colab notebooks. We use standard `plt.show()`.
- **Channel Modeling (`src/utils/channels.py`):** Fading is implemented via complex multiplication `(I + jQ) * h`. `h` is generated using numpy for vectorized speed over batches of `(N, 2, 128)`.

## 5. Next Steps (Current To-Do)
1. Generate faded datasets (Rayleigh, Rician) using `src/utils/channels.py`.
2. Visualize the effect of fading on the IQ signals (e.g., in a new notebook `03_fading_dataset_generation.ipynb`).
3. Train both MCLDNN and PET-CGDNN on the faded datasets to observe performance degradation.
