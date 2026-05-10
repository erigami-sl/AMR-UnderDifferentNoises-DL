"""
AMR-UnderDifferentNoises-DL — Colab Tek Dosya (All-in-One)
===========================================================
Bu dosya, Google Colab'da projeyi çalıştırmak için gereken tüm kaynak kodlarını
tek bir dosyada birleştirilmiş halde içerir.

İçerik:
    1. Dataset Loader  (src/utils/dataset.py)
    2. MCLDNN Model    (src/models/mcldnn.py)
    3. PET-CGDNN Model (src/models/petcgdnn.py)
    4. Metrics & Plots (src/utils/metrics.py)

Kullanım (Colab'da):
    from amr_all_in_one_core_toolkit import *
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import pickle
import time
import shutil

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, Conv2D, Dropout,
    concatenate, Flatten, Reshape, LSTM,
    Activation, Lambda, Multiply, Add, Subtract, GRU
)
from sklearn.metrics import f1_score, matthews_corrcoef


# =============================================================================
# 1. DATASET LOADER
# =============================================================================

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
    Xd = pickle.load(open(filename, 'rb'), encoding='iso-8859-1')

    mods, snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0, 1]]

    X = []
    lbl = []
    train_idx = []
    val_idx = []
    np.random.seed(seed)
    a = 0

    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod, snr)])
            for i in range(Xd[(mod, snr)].shape[0]):
                lbl.append((mod, snr))
            train_idx += list(np.random.choice(
                range(a * 1000, (a + 1) * 1000), size=600, replace=False
            ))
            val_idx += list(np.random.choice(
                list(set(range(a * 1000, (a + 1) * 1000)) - set(train_idx)),
                size=200, replace=False
            ))
            a += 1

    X = np.vstack(X)

    n_examples = X.shape[0]
    test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(val_idx))
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

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
    """
    def _format(X):
        X1 = np.expand_dims(X[:, 0, :], axis=2)
        X2 = np.expand_dims(X[:, 1, :], axis=2)
        X_iq = np.expand_dims(X, axis=3)
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
    """
    def _format(X):
        X_swapped = X.swapaxes(2, 1)
        X1 = X_swapped[:, :, 0]
        X2 = X_swapped[:, :, 1]
        X_iq = np.expand_dims(X_swapped, axis=3)
        return [X_iq, X1, X2]

    return {
        'train': _format(X_train),
        'val': _format(X_val),
        'test': _format(X_test)
    }


# =============================================================================
# 2. MCLDNN MODEL
# =============================================================================

def MCLDNN(weights=None,
           input_shape1=[2, 128],
           input_shape2=[128, 1],
           classes=11,
           **kwargs):
    """
    Build the MCLDNN model.
    A Spatiotemporal Multi-Channel Learning Framework for Automatic Modulation Recognition.

    Reference:
        Xu, J., Luo, C., Parr, G., & Luo, Y. (2020).
        IEEE Wireless Communications Letters, 9(10), 1629-1632.
    """
    if weights is not None and not os.path.exists(weights):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.5

    # --- Inputs ---
    input1 = Input(input_shape1 + [1], name='input1')
    input2 = Input(input_shape2, name='input2')
    input3 = Input(input_shape2, name='input3')

    # --- Separate Channel Convolutional Branches ---
    x1 = Conv2D(50, (2, 8), padding='same', activation='relu',
                name='conv1_1', kernel_initializer='glorot_uniform')(input1)

    x2 = Conv1D(50, 8, padding='causal', activation='relu',
                name='conv1_2', kernel_initializer='glorot_uniform')(input2)
    x2_reshape = Reshape([-1, 128, 50], name='reshape_i')(x2)

    x3 = Conv1D(50, 8, padding='causal', activation='relu',
                name='conv1_3', kernel_initializer='glorot_uniform')(input3)
    x3_reshape = Reshape([-1, 128, 50], name='reshape_q')(x3)

    # --- Concatenation and Fusion ---
    x = concatenate([x2_reshape, x3_reshape], axis=1)
    x = Conv2D(50, (1, 8), padding='same', activation='relu',
               name='conv2', kernel_initializer='glorot_uniform')(x)
    x = concatenate([x1, x])
    x = Conv2D(100, (2, 5), padding='valid', activation='relu',
               name='conv4', kernel_initializer='glorot_uniform')(x)

    # --- LSTM Unit ---
    x = Reshape(target_shape=(124, 100), name='reshape_lstm')(x)
    x = LSTM(units=128, return_sequences=True, name='lstm1')(x)
    x = LSTM(units=128, name='lstm2')(x)

    # --- Dense Classifier ---
    x = Dense(128, activation='selu', name='fc1')(x)
    x = Dropout(dr, name='dropout1')(x)
    x = Dense(128, activation='selu', name='fc2')(x)
    x = Dropout(dr, name='dropout2')(x)
    x = Dense(classes, activation='softmax', name='softmax')(x)

    model = Model(inputs=[input1, input2, input3], outputs=x)

    if weights is not None:
        model.load_weights(weights)

    return model


# =============================================================================
# 3. PET-CGDNN MODEL
# =============================================================================

def _cos(x):
    """Cosine activation for phase estimation."""
    return tf.math.cos(x)


def _sin(x):
    """Sine activation for phase estimation."""
    return tf.math.sin(x)


def PETCGDNN(weights=None,
             input_shape=[128, 2],
             input_shape2=[128],
             classes=11,
             **kwargs):
    """
    Build the PET-CGDNN model.
    An Efficient Deep Learning Model for Automatic Modulation Recognition
    Based on Parameter Estimation and Transformation.

    Reference:
        Zhang, F., Luo, C., Xu, J., & Luo, Y. (2021).
        IEEE Communications Letters, 25(10), 3287-3291.
    """
    if weights is not None and not os.path.exists(weights):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.5

    # --- Inputs ---
    inp = Input(input_shape + [1], name='input1')
    input1 = Input(input_shape2, name='input2')
    input2 = Input(input_shape2, name='input3')

    # --- Parameter Estimation (Phase theta) ---
    x1 = Flatten(name='flatten_iq')(inp)
    x1 = Dense(1, name='phase_estimate')(x1)
    x1 = Activation('linear', name='phase_linear')(x1)

    # --- Phase Transformation ---
    cos1 = Lambda(_cos, name='lambda_cos')(x1)
    sin1 = Lambda(_sin, name='lambda_sin')(x1)

    # y1 = I*cos(theta) + Q*sin(theta)
    x11 = Multiply(name='mul_i_cos')([input1, cos1])
    x12 = Multiply(name='mul_q_sin')([input2, sin1])
    y1 = Add(name='add_y1')([x11, x12])

    # y2 = Q*cos(theta) - I*sin(theta)
    x21 = Multiply(name='mul_q_cos')([input2, cos1])
    x22 = Multiply(name='mul_i_sin')([input1, sin1])
    y2 = Subtract(name='sub_y2')([x21, x22])

    y1 = Reshape(target_shape=(128, 1), name='reshape1')(y1)
    y2 = Reshape(target_shape=(128, 1), name='reshape2')(y2)
    x11 = concatenate([y1, y2])
    x3 = Reshape(target_shape=(128, 2, 1), name='reshape3')(x11)

    # --- Spatial Feature Extraction (Conv2D) ---
    x3 = Conv2D(75, (8, 2), padding='valid', activation='relu',
                name='conv1_1', kernel_initializer='glorot_uniform')(x3)
    x3 = Conv2D(25, (5, 1), padding='valid', activation='relu',
                name='conv1_2', kernel_initializer='glorot_uniform')(x3)

    # --- Temporal Feature Extraction (GRU) ---
    x4 = Reshape(target_shape=(117, 25), name='reshape4')(x3)
    x4 = GRU(units=128, name='gru')(x4)

    # --- Classifier ---
    x = Dense(classes, activation='softmax', name='softmax')(x4)

    model = Model(inputs=[inp, input1, input2], outputs=x)

    if weights is not None:
        model.load_weights(weights)

    return model


# =============================================================================
# 4. METRICS & PLOTS
# =============================================================================

def calculate_confusion_matrix(Y, Y_hat, classes):
    """
    Calculate the normalized confusion matrix.
    """
    n_classes = len(classes)
    conf = np.zeros([n_classes, n_classes])

    for k in range(0, Y.shape[0]):
        i = list(Y[k, :]).index(1)
        j = int(np.argmax(Y_hat[k, :]))
        conf[i, j] = conf[i, j] + 1

    confnorm = np.zeros([n_classes, n_classes])
    for i in range(0, n_classes):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])

    right = np.sum(np.diag(conf))
    wrong = np.sum(conf) - right
    return confnorm, right, wrong


def plot_confusion_matrix(cm, labels, title='Confusion Matrix',
                          cmap=plt.get_cmap("Blues"), save_filename=None):
    """Plot and optionally save a confusion matrix."""
    plt.figure(figsize=(4, 3), dpi=600)
    plt.imshow(cm * 100, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90, size=12)
    plt.yticks(tick_marks, labels, size=12)

    for i in range(len(tick_marks)):
        for j in range(len(tick_marks)):
            val = int(np.around(cm[i, j] * 100))
            if i != j:
                plt.text(j, i, val, ha="center", va="center", fontsize=10)
            else:
                fontsize = 7 if val == 100 else 10
                plt.text(j, i, val, ha="center", va="center",
                         fontsize=fontsize, color='darkorange')

    plt.tight_layout()
    if save_filename is not None:
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        plt.savefig(save_filename, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_snr_accuracy(acc, snrs, title='Classification Accuracy on RadioML 2016.10a',
                      save_filename=None):
    """Plot SNR vs Classification Accuracy curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(snrs, list(map(lambda x: acc[x], snrs)), 'o-', linewidth=2)
    plt.xlabel("Signal to Noise Ratio (dB)", fontsize=12)
    plt.ylabel("Classification Accuracy", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_filename is not None:
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_per_mod_accuracy(acc_mod_snr, snrs, classes, save_filename=None):
    """Plot per-modulation accuracy across SNR levels."""
    plt.figure(figsize=(12, 8))
    plt.xlabel("Signal to Noise Ratio (dB)", fontsize=12)
    plt.ylabel("Classification Accuracy", fontsize=12)
    plt.title("Classification Accuracy per Modulation Type", fontsize=14)

    for i in range(len(classes)):
        plt.plot(snrs, acc_mod_snr[i], 'o-', label=classes[i], linewidth=1.5, markersize=4)

    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_filename is not None:
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_snr_f1(f1_scores, snrs, title='F1 Score (Macro) vs SNR',
                save_filename=None):
    """Plot SNR vs F1 Score (macro average) curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(snrs, list(map(lambda x: f1_scores[x], snrs)), 's-',
             linewidth=2, color='#E91E63', label='F1 (macro)')
    plt.xlabel("Signal to Noise Ratio (dB)", fontsize=12)
    plt.ylabel("F1 Score (Macro)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    if save_filename is not None:
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_snr_mcc(mcc_scores, snrs, title='MCC (Matthews Correlation Coefficient) vs SNR',
                 save_filename=None):
    """Plot SNR vs Matthews Correlation Coefficient (MCC) curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(snrs, list(map(lambda x: mcc_scores[x], snrs)), 'D-',
             linewidth=2, color='#9C27B0', label='MCC')
    plt.xlabel("Signal to Noise Ratio (dB)", fontsize=12)
    plt.ylabel("MCC", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([-1, 1])
    plt.tight_layout()
    if save_filename is not None:
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_training_history(history, save_dir=None):
    """Plot training and validation loss/accuracy curves."""
    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.title('Training Loss')
    plt.plot(history.epoch, history.history['loss'], label='Train Loss')
    plt.plot(history.epoch, history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_loss.png'), dpi=300)
    plt.show()
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(10, 5))
    plt.title('Training Accuracy')
    acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
    plt.plot(history.epoch, history.history[acc_key], label='Train Accuracy')
    plt.plot(history.epoch, history.history[val_acc_key], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'training_accuracy.png'), dpi=300)
    plt.show()
    plt.close()


def evaluate_model(model, X_test_inputs, Y_test, lbl, test_idx, snrs, classes,
                   batch_size=400, results_dir=None):
    """
    Full evaluation pipeline: confusion matrices, SNR vs accuracy,
    per-mod accuracy, F1 Score, MCC.

    Returns
    -------
    acc : dict
        SNR -> accuracy mapping
    acc_mod_snr : ndarray
        Per-modulation, per-SNR accuracy matrix
    f1_scores : dict
        SNR -> F1 Score (macro) mapping
    mcc_scores : dict
        SNR -> MCC mapping
    """
    fig_dir = os.path.join(results_dir, 'figures') if results_dir else None
    pred_dir = os.path.join(results_dir, 'predictions') if results_dir else None

    # Overall confusion matrix
    test_Y_hat = model.predict(X_test_inputs, batch_size=batch_size)
    confnorm, _, _ = calculate_confusion_matrix(Y_test, test_Y_hat, classes)
    if fig_dir:
        plot_confusion_matrix(
            confnorm, labels=classes,
            save_filename=os.path.join(fig_dir, 'confusion_matrix_overall.png')
        )

    # Per-SNR evaluation
    acc = {}
    f1_scores = {}
    mcc_scores = {}
    acc_mod_snr = np.zeros((len(classes), len(snrs)))
    test_SNRs = [lbl[x][1] for x in test_idx]

    for i, snr in enumerate(snrs):
        snr_mask = np.where(np.array(test_SNRs) == snr)

        test_X_i = [inp[snr_mask] for inp in X_test_inputs]
        test_Y_i = Y_test[snr_mask]

        test_Y_i_hat = model.predict(test_X_i)
        confnorm_i, cor, ncor = calculate_confusion_matrix(test_Y_i, test_Y_i_hat, classes)

        acc[snr] = 1.0 * cor / (cor + ncor)
        acc_mod_snr[:, i] = np.round(
            np.diag(confnorm_i) / np.sum(confnorm_i, axis=1), 3
        )

        # F1 Score (macro) and MCC for this SNR
        y_true = np.argmax(test_Y_i, axis=1)
        y_pred = np.argmax(test_Y_i_hat, axis=1)
        f1_scores[snr] = f1_score(y_true, y_pred, average='macro')
        mcc_scores[snr] = matthews_corrcoef(y_true, y_pred)

        # Save per-SNR confusion matrix
        if fig_dir:
            plot_confusion_matrix(
                confnorm_i, labels=classes,
                title=f"Confusion Matrix (SNR={snr}dB)",
                save_filename=os.path.join(
                    fig_dir, f'confusion_snr_{snr:+d}dB_acc_{100*acc[snr]:.1f}.png'
                )
            )

    # Plot SNR vs accuracy, F1, MCC
    if fig_dir:
        plot_snr_accuracy(acc, snrs, save_filename=os.path.join(fig_dir, 'snr_vs_accuracy.png'))
        plot_per_mod_accuracy(acc_mod_snr, snrs, classes,
                              save_filename=os.path.join(fig_dir, 'per_mod_accuracy.png'))
        plot_snr_f1(f1_scores, snrs,
                    save_filename=os.path.join(fig_dir, 'snr_vs_f1_macro.png'))
        plot_snr_mcc(mcc_scores, snrs,
                     save_filename=os.path.join(fig_dir, 'snr_vs_mcc_metric.png'))

    # Save results
    if pred_dir:
        os.makedirs(pred_dir, exist_ok=True)
        with open(os.path.join(pred_dir, 'acc.pkl'), 'wb') as f:
            pickle.dump(acc, f)
        with open(os.path.join(pred_dir, 'acc_mod_snr.pkl'), 'wb') as f:
            pickle.dump(acc_mod_snr, f)
        with open(os.path.join(pred_dir, 'f1_macro_scores.pkl'), 'wb') as f:
            pickle.dump(f1_scores, f)
        with open(os.path.join(pred_dir, 'mcc_metric_scores.pkl'), 'wb') as f:
            pickle.dump(mcc_scores, f)

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for snr in snrs:
        print(f"  SNR {snr:+3d} dB : Acc={acc[snr]*100:5.2f}%  F1={f1_scores[snr]*100:5.2f}%  MCC={mcc_scores[snr]:.4f}")
    overall_acc = np.mean(list(acc.values()))
    overall_f1 = np.mean(list(f1_scores.values()))
    overall_mcc = np.mean(list(mcc_scores.values()))
    print(f"\n  Overall Average Accuracy : {overall_acc*100:.2f}%")
    print(f"  Average F1 (macro)      : {overall_f1*100:.2f}%")
    print(f"  Average MCC             : {overall_mcc:.4f}")
    print("=" * 50)

    return acc, acc_mod_snr, f1_scores, mcc_scores
