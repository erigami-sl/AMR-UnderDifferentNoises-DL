# Deep Learning-Based AMR Under Different SISO Channel Conditions

Automatic Modulation Recognition (AMR) for low-SNR signals under various SISO channel conditions (AWGN, Rayleigh, Rician fading).

Based on: Zhang, F. et al. (2022). *"Deep Learning Based Automatic Modulation Recognition: Models, Datasets, and Challenges"*, Digital Signal Processing. [DOI](https://doi.org/10.1016/j.dsp.2022.103650)

## Models

| Model | Type | Paper | Year |
|-------|------|-------|------|
| **MCLDNN** | Multi-Channel CNN + LSTM | [IEEE 2020](https://ieeexplore.ieee.org/abstract/document/9106397) | 2020 |
| **PET-CGDNN** | Phase Estimation + CNN + GRU | [IEEE 2021](https://ieeexplore.ieee.org/abstract/document/9507514) | 2021 |

## Dataset

**RadioML 2016.10a** — 11 modulation classes, 220,000 samples, 2×128 IQ format, SNR: -20 to +18 dB.

## Project Structure

```
├── src/
│   ├── config.py              # Central configuration
│   ├── utils/
│   │   ├── dataset.py         # Data loading & preprocessing
│   │   └── metrics.py         # Evaluation & plotting
│   └── models/
│       ├── mcldnn.py           # MCLDNN architecture
│       └── petcgdnn.py         # PET-CGDNN architecture
├── notebooks/                  # Colab notebooks
├── data/                       # Dataset (not tracked by git)
├── results/                    # Training results
├── RML201610a/                 # Original benchmark code (reference)
│   ├── MCLDNN/
│   └── PET-CGDNN/
├── requirements.txt
└── README.md
```

## Setup (Google Colab)

```python
# Mount Drive and install dependencies
from google.colab import drive
drive.mount('/content/drive')
!pip install commpy
```

## References

- Baseline: [AMR-Benchmark](https://github.com/Richardzhangxx/AMR-Benchmark)
- Channel Models: [CommPy](https://github.com/veeresht/CommPy)
- Dataset: [RadioML](http://radioml.com)

## Citation

```bibtex
@article{ZHANG2022103650,
    title={Deep Learning Based Automatic Modulation Recognition: Models, Datasets, and Challenges},
    author={Fuxin Zhang and Chunbo Luo and Jialang Xu and Yang Luo and FuChun Zheng},
    journal={Digital Signal Processing},
    year={2022},
    doi={https://doi.org/10.1016/j.dsp.2022.103650}
}
```
