"""
Dataset Loader for RML2016.10a
==============================
Centralized data loading and preprocessing for both MCLDNN and PET-CGDNN models.
Merged from the original per-model dataset2016.py files.

Dataset format:
    - Pickle dict with keys: (modulation_type, snr)
    - Values: ndarray of shape (1000, 2, 128)
    - 11 modulation classes × 20 SNR levels = 220,000 total samples
    - Split: 60% train, 20% val, 20% test (per mod/snr combination)
"""

import pickle
import numpy as np


def load_data(filename, seed=2016):
    """
    Load and split the RML2016.10a dataset.

    Parameters
    ----------
    filename : str
        Path to RML2016.10a_dict.pkl
    seed : int
        Random seed for reproducibility (default: 2016, same as original paper)

    Returns
    -------
    (mods, snrs, lbl) : tuple
        Modulation types list, SNR values list, and per-sample labels
    (X_train, Y_train) : tuple
        Training data and one-hot labels
    (X_val, Y_val) : tuple
        Validation data and one-hot labels
    (X_test, Y_test) : tuple
        Test data and one-hot labels
    (train_idx, val_idx, test_idx) : tuple
        Index arrays for each split
    """
    # Load pickle file
    Xd = pickle.load(open(filename, 'rb'), encoding='iso-8859-1')

    # Extract sorted modulation types and SNR values
    mods, snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0, 1]]

    X = []
    lbl = []
    train_idx = []
    val_idx = []
    np.random.seed(seed)
    a = 0

    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod, snr)])  # ndarray(1000, 2, 128)
            for i in range(Xd[(mod, snr)].shape[0]):
                lbl.append((mod, snr))
            # 60/20/20 split per mod/snr combination
            train_idx += list(np.random.choice(
                range(a * 1000, (a + 1) * 1000), size=600, replace=False
            ))
            val_idx += list(np.random.choice(
                list(set(range(a * 1000, (a + 1) * 1000)) - set(train_idx)),
                size=200, replace=False
            ))
            a += 1

    # Stack all samples: (220000, 2, 128)
    X = np.vstack(X)

    n_examples = X.shape[0]
    test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(val_idx))
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    # One-hot encoding
    def to_onehot(yy):
        yy1 = np.zeros([len(yy), len(mods)])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1

    Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    Y_val = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), val_idx)))
    Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

    return (mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), \
           (X_test, Y_test), (train_idx, val_idx, test_idx)


def prepare_data_mcldnn(X_train, X_val, X_test):
    """
    Prepare data in MCLDNN format.
    MCLDNN expects 3 inputs:
      - input1: IQ combined (N, 2, 128, 1)
      - input2: I channel only (N, 128, 1)
      - input3: Q channel only (N, 128, 1)

    Parameters
    ----------
    X_train, X_val, X_test : ndarray
        Raw data in shape (N, 2, 128)

    Returns
    -------
    dict with keys 'train', 'val', 'test', each containing [input1, input2, input3]
    """
    def _format(X):
        X1 = np.expand_dims(X[:, 0, :], axis=2)  # I channel: (N, 128, 1)
        X2 = np.expand_dims(X[:, 1, :], axis=2)  # Q channel: (N, 128, 1)
        X_iq = np.expand_dims(X, axis=3)          # IQ combined: (N, 2, 128, 1)
        return [X_iq, X1, X2]

    return {
        'train': _format(X_train),
        'val': _format(X_val),
        'test': _format(X_test)
    }


def prepare_data_petcgdnn(X_train, X_val, X_test):
    """
    Prepare data in PET-CGDNN format.
    PET-CGDNN expects 3 inputs:
      - input1: IQ combined, transposed (N, 128, 2, 1)
      - input2: I channel (N, 128)
      - input3: Q channel (N, 128)

    NOTE: PET-CGDNN swaps axes from (N, 2, 128) to (N, 128, 2) before processing.

    Parameters
    ----------
    X_train, X_val, X_test : ndarray
        Raw data in shape (N, 2, 128)

    Returns
    -------
    dict with keys 'train', 'val', 'test', each containing [input1, input2, input3]
    """
    def _format(X):
        X_swapped = X.swapaxes(2, 1)              # (N, 2, 128) → (N, 128, 2)
        X1 = X_swapped[:, :, 0]                   # I channel: (N, 128)
        X2 = X_swapped[:, :, 1]                    # Q channel: (N, 128)
        X_iq = np.expand_dims(X_swapped, axis=3)  # IQ combined: (N, 128, 2, 1)
        return [X_iq, X1, X2]

    return {
        'train': _format(X_train),
        'val': _format(X_val),
        'test': _format(X_test)
    }


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '..')
    from config import get_dataset_path

    path = get_dataset_path()
    print(f"Loading dataset from: {path}")
    (mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), \
        (X_test, Y_test), (train_idx, val_idx, test_idx) = load_data(path)

    print(f"\nModulations: {mods}")
    print(f"SNR range: {snrs[0]} to {snrs[-1]} dB")
    print(f"\nTrain: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Val:   X={X_val.shape},   Y={Y_val.shape}")
    print(f"Test:  X={X_test.shape},  Y={Y_test.shape}")

    # Test MCLDNN format
    mcldnn_data = prepare_data_mcldnn(X_train, X_val, X_test)
    print(f"\nMCLDNN format:")
    print(f"  input1 (IQ): {mcldnn_data['train'][0].shape}")
    print(f"  input2 (I):  {mcldnn_data['train'][1].shape}")
    print(f"  input3 (Q):  {mcldnn_data['train'][2].shape}")

    # Test PET-CGDNN format
    pet_data = prepare_data_petcgdnn(X_train, X_val, X_test)
    print(f"\nPET-CGDNN format:")
    print(f"  input1 (IQ): {pet_data['train'][0].shape}")
    print(f"  input2 (I):  {pet_data['train'][1].shape}")
    print(f"  input3 (Q):  {pet_data['train'][2].shape}")
