"""
Project Configuration
=====================
Central configuration for paths, hyperparameters, and constants.
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Dataset filename (should be placed inside DATA_DIR)
DATASET_FILENAME = 'RML2016.10a_dict.pkl'
DATASET_PATH = os.path.join(DATA_DIR, DATASET_FILENAME)

# Google Colab paths (used when running on Colab)
COLAB_DRIVE_PATH = '/content/drive/MyDrive/AMR-Project'
COLAB_DATASET_PATH = os.path.join(COLAB_DRIVE_PATH, DATASET_FILENAME)

# ──────────────────────────────────────────────
# Dataset Constants
# ──────────────────────────────────────────────
# These are the actual key names in the RML2016.10a pickle file (sorted order)
MODULATION_CLASSES = [
    '8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK',
    'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM'
]
NUM_CLASSES = len(MODULATION_CLASSES)  # 11

# Display names for plots (prettier labels used in the original paper)
MODULATION_DISPLAY_NAMES = {
    'PAM4': '4-PAM', 'QAM16': '16-QAM', 'QAM64': '64-QAM'
}
SAMPLE_LENGTH = 128                     # IQ sample length
NUM_CHANNELS = 2                        # I and Q channels
RANDOM_SEED = 2016                      # For reproducibility (same as original paper)

# Train/Val/Test split sizes (per modulation/SNR combination, out of 1000)
TRAIN_SIZE = 600
VAL_SIZE = 200
TEST_SIZE = 200  # remainder

# ──────────────────────────────────────────────
# Training Hyperparameters
# ──────────────────────────────────────────────
EPOCHS = 1000
BATCH_SIZE = 400
LEARNING_RATE = 0.001
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 5
LR_MIN = 1e-7
EARLY_STOP_PATIENCE = 50

# ──────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────
def get_dataset_path():
    """Returns the appropriate dataset path based on environment."""
    # Check Colab first
    if os.path.exists('/content'):
        if os.path.exists(COLAB_DATASET_PATH):
            return COLAB_DATASET_PATH
    # Local path
    if os.path.exists(DATASET_PATH):
        return DATASET_PATH
    raise FileNotFoundError(
        f"Dataset not found. Looked in:\n"
        f"  Local: {DATASET_PATH}\n"
        f"  Colab: {COLAB_DATASET_PATH}\n"
        f"Please download RML2016.10a_dict.pkl and place it in the data/ directory."
    )

def get_results_dir(model_name, channel_type='awgn'):
    """Returns and creates the results directory for a given model/channel combination."""
    result_path = os.path.join(RESULTS_DIR, model_name, channel_type)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(os.path.join(result_path, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(result_path, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(result_path, 'predictions'), exist_ok=True)
    return result_path

def is_colab():
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False
