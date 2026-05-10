# Deep Learning-Based AMR Under Different SISO Channel Conditions

## 1. Project Title and Team Members
**Project Title:** Automatic Modulation Recognition (AMR) Under Different SISO Channel Conditions Using Deep Learning  
**Team Members:** 
- Batuhan [Soyadınızı Ekleyin] (Geliştirici)
- [Varsa Diğer Takım Üyeleri]

## 2. Brief Project Description
Automatic Modulation Recognition (AMR) is a crucial component in modern wireless communication and cognitive radio systems. This project explores the performance of two prominent Deep Learning architectures (MCLDNN and PET-CGDNN) for classifying 11 modulation schemes across a wide range of Signal-to-Noise Ratios (SNR: -20 dB to +18 dB). 

We extend the baseline AWGN-only analysis by introducing complex multipath fading channels (Rayleigh and Rician K=3, K=10) to simulate real-world environmental signal degradation and evaluate the models' robustness using advanced metrics like F1-Macro and MCC.

## 3. Link to Original Paper and GitHub Repository
- **Original Paper:** F. Zhang et al., *"Deep Learning Based Automatic Modulation Recognition: Models, Datasets, and Challenges"*, Digital Signal Processing, 2022. [DOI: 10.1016/j.dsp.2022.103650](https://doi.org/10.1016/j.dsp.2022.103650)
- **Original Baseline Repository:** [AMR-Benchmark](https://github.com/Richardzhangxx/AMR-Benchmark)
- **Our Improved Project Repository:** [AMR-UnderDifferentNoises-DL](https://github.com/erigami-sl/AMR-UnderDifferentNoises-DL)

## 4. Summary of Our Modifications and Improvements
While the original paper focused primarily on AWGN channels with basic Accuracy metrics, our project introduces several significant enhancements:
- **TensorFlow 2.x Migration:** Completely migrated the legacy Keras 2.2 / TF 1.x codebase to modern TensorFlow 2.x (`tf.keras`) ensuring future compatibility.
- **SISO Channel Simulations:** Developed custom mathematical pipelines (`src/utils/channels.py`) to generate Rayleigh and Rician (K=3, K=10) fading effects, directly applying them to the RadioML IQ dataset.
- **Advanced Evaluation Metrics:** Integrated F1-Score (Macro) and Matthews Correlation Coefficient (MCC) alongside Accuracy and Confusion Matrices for deeper statistical analysis.
- **Automated Fine-Tuning Pipeline:** Created automated scripts to fine-tune AWGN pre-trained models on fading channels to improve real-world recovery.
- **Consolidated Toolkit:** Merged all utility functions into a single `amr_all_in_one_core_toolkit.py` file to enable one-click deployments on Google Colab without complex Git setups.
- **Visualization Suite:** Developed automated comparison graphing tools for Model vs Model and Channel vs Channel evaluations.

## 5. Requirements and Installation Instructions
This project is optimized for **Google Colab** with a GPU (T4 recommended).  
For **Local Python 3.8+ Environment:**
```bash
git clone https://github.com/erigami-sl/AMR-UnderDifferentNoises-DL.git
cd AMR-UnderDifferentNoises-DL
pip install -r requirements.txt
```
*(Key dependencies: `tensorflow>=2.10`, `numpy`, `matplotlib`, `scikit-learn`, `commpy`, `scipy`)*

## 6. Dataset Download/Preparation Instructions
We utilize the **RadioML 2016.10a** dataset. Due to size limitations (~611MB), it is not tracked in this GitHub repository.
1. Download the dataset from the official [RadioML website](http://radioml.com).
2. **For Google Colab:** Upload the `RML2016.10a_dict.pkl` file to your Google Drive under: `MyDrive/AMR-Project/RML2016.10a_dict.pkl`
3. **For Local Execution:** Place it inside the `data/` directory of this project.

## 7. How to Reproduce Original Results
To replicate the original AWGN baseline results:
1. Open Google Colab.
2. Load and run `notebooks/01_baseline_mcldnn.ipynb` (for MCLDNN) and `notebooks/02_baseline_petcgdnn.ipynb` (for PET-CGDNN).
3. The notebooks will automatically mount your drive, load the dataset, train the models using TF2, and output the original Accuracy baseline charts.

## 8. How to Run Our Improved Version
1. **Generate Faded Datasets:** Run `notebooks/04_generate_faded_datasets.ipynb`. This will apply Rayleigh and Rician fading equations to the AWGN dataset and save the new `.pkl` datasets to your Drive.
2. **Train Models on Fading Channels:** Run `notebooks/06_train_petcgdnn_fading.ipynb` and `07_train_mcldnn_fading.ipynb` to train models on the newly generated datasets.
3. **Fine-Tuning:** Use `notebooks/08_finetuning_awgn_on_faded.ipynb` to take baseline pre-trained models and adapt them to fading channels.
4. **Analysis & Comparison:** Execute `notebooks/09_results_analysis_and_comparison.ipynb`. This will automatically parse your `.pkl` history files and generate high-resolution comparative graphs (saved to `fine_tuning_results/Karsilastirma_Grafikleri/`).

## 9. Comparison of Results (Original vs Ours)
*(Values are indicative based on maximum SNR (+18dB) performance trends observed)*

| Metric / Scenario | Original Baseline (AWGN) | Our Rayleigh | Our Rician K=3 | Our Rician K=10 |
|-------------------|--------------------------|--------------|----------------|-----------------|
| **MCLDNN Max Accuracy** | ~92% | *Significant drop expected* | *Moderate drop* | *Close to AWGN* |
| **PET-CGDNN Max Acc** | ~90% | *Better phase recovery* | *Moderate drop* | *Close to AWGN* |
| **Evaluation Depth** | Accuracy Only | Acc, F1, MCC | Acc, F1, MCC | Acc, F1, MCC |

*Note: You can view the exact empirical charts generated by Notebook 09 in your Google Drive or run the evaluation cells to print the real-time statistical summaries.*

## 10. File Structure Explanation
```text
AMR-UnderDifferentNoises-DL/
├── .gitignore
├── CHANGELOG.md                # Detailed task history and sprint updates
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── amr_all_in_one_core_toolkit.py # Unified script for Colab use
├── data/                       # Dataset folder (Ignored by Git)
├── results/                    # Original baseline training outputs
├── notebooks/                  # Interactive Colab Environment
│   ├── 01_baseline_mcldnn.ipynb
│   ├── 02_baseline_petcgdnn.ipynb
│   ├── 03_verify_channels.ipynb
│   ├── 04_generate_faded_datasets.ipynb
│   ├── 05_visualize_fading_effects.ipynb
│   ├── 06_train_petcgdnn_fading.ipynb
│   ├── 07_train_mcldnn_fading.ipynb
│   ├── 08_finetuning_awgn_on_faded.ipynb
│   └── 09_results_analysis_and_comparison.ipynb
└── src/                        # Core source code modules
    ├── config.py
    ├── models/                 # Neural Network Architectures (TF2)
    └── utils/                  # Channels, dataset loaders, metrics
```

## 11. Expected Outputs and Runtime
- **Runtime:** Training one model (e.g., MCLDNN) on the full dataset (220,000 samples) for 50-100 epochs takes approximately **1 to 2 hours** on a Google Colab T4 GPU.
- **Outputs:**
  - `history.pkl` (Training loss/accuracy per epoch)
  - `best.weights.h5` (Best model weights for Keras 3 compatibility)
  - `predictions/` folder containing `acc.pkl`, `f1_macro_scores.pkl`, `mcc_metric_scores.pkl`
  - High-resolution `confusion_matrix.png` and comparative charts (Model vs Model, Channel vs Channel).
