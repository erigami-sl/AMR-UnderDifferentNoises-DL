"""
Evaluation Metrics and Plotting Utilities
==========================================
Centralized evaluation tools merged from both models' mltools.py files.
Provides confusion matrix computation, accuracy plotting, and SNR analysis.

Adapted from the code (https://github.com/leena201818/radiom) contributed by leena201818.
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


def calculate_confusion_matrix(Y, Y_hat, classes):
    """
    Calculate the normalized confusion matrix.

    Parameters
    ----------
    Y : ndarray (N, num_classes)
        One-hot encoded true labels
    Y_hat : ndarray (N, num_classes)
        Model predictions (probabilities)
    classes : list
        List of class names

    Returns
    -------
    confnorm : ndarray
        Normalized confusion matrix (row-normalized)
    right : int
        Number of correct predictions
    wrong : int
        Number of incorrect predictions
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
    """
    Plot and optionally save a confusion matrix.

    Parameters
    ----------
    cm : ndarray
        Normalized confusion matrix (values 0-1)
    labels : list of str
        Class labels for axes
    title : str
        Plot title
    cmap : colormap
        Matplotlib colormap
    save_filename : str or None
        If provided, saves figure to this path
    """
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
    """
    Plot SNR vs Classification Accuracy curve.

    Parameters
    ----------
    acc : dict
        Dictionary mapping SNR values to accuracy values
    snrs : list
        List of SNR values
    title : str
        Plot title
    save_filename : str or None
        If provided, saves figure to this path
    """
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
    """
    Plot per-modulation accuracy across SNR levels.

    Parameters
    ----------
    acc_mod_snr : ndarray (num_classes, num_snrs)
        Accuracy for each modulation at each SNR
    snrs : list
        List of SNR values
    classes : list
        List of modulation class names
    save_filename : str or None
        If provided, saves figure to this path
    """
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


def plot_training_history(history, save_dir=None):
    """
    Plot training and validation loss/accuracy curves.

    Parameters
    ----------
    history : keras History object
        Training history
    save_dir : str or None
        Directory to save figures
    """
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
    # TF2 uses 'accuracy' instead of 'acc'
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
    Full evaluation pipeline: confusion matrices, SNR vs accuracy, per-mod accuracy.

    Parameters
    ----------
    model : keras Model
        Trained model
    X_test_inputs : list of ndarray
        Model inputs for test data (format-specific)
    Y_test : ndarray
        One-hot test labels
    lbl : list
        Per-sample (modulation, snr) labels
    test_idx : list
        Test sample indices
    snrs : list
        SNR values
    classes : list
        Modulation class names
    batch_size : int
        Prediction batch size
    results_dir : str or None
        Directory to save results

    Returns
    -------
    acc : dict
        SNR → accuracy mapping
    acc_mod_snr : ndarray
        Per-modulation, per-SNR accuracy matrix
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
    acc_mod_snr = np.zeros((len(classes), len(snrs)))
    test_SNRs = [lbl[x][1] for x in test_idx]

    for i, snr in enumerate(snrs):
        snr_mask = np.where(np.array(test_SNRs) == snr)

        # Extract test data at this SNR
        test_X_i = [inp[snr_mask] for inp in X_test_inputs]
        test_Y_i = Y_test[snr_mask]

        # Predict
        test_Y_i_hat = model.predict(test_X_i)
        confnorm_i, cor, ncor = calculate_confusion_matrix(test_Y_i, test_Y_i_hat, classes)

        acc[snr] = 1.0 * cor / (cor + ncor)
        acc_mod_snr[:, i] = np.round(
            np.diag(confnorm_i) / np.sum(confnorm_i, axis=1), 3
        )

        # Save per-SNR confusion matrix
        if fig_dir:
            plot_confusion_matrix(
                confnorm_i, labels=classes,
                title=f"Confusion Matrix (SNR={snr}dB)",
                save_filename=os.path.join(
                    fig_dir, f'confusion_snr_{snr:+d}dB_acc_{100*acc[snr]:.1f}.png'
                )
            )

    # Plot SNR vs accuracy
    if fig_dir:
        plot_snr_accuracy(acc, snrs, save_filename=os.path.join(fig_dir, 'snr_vs_accuracy.png'))
        plot_per_mod_accuracy(acc_mod_snr, snrs, classes,
                              save_filename=os.path.join(fig_dir, 'per_mod_accuracy.png'))

    # Save results
    if pred_dir:
        os.makedirs(pred_dir, exist_ok=True)
        with open(os.path.join(pred_dir, 'acc.pkl'), 'wb') as f:
            pickle.dump(acc, f)
        with open(os.path.join(pred_dir, 'acc_mod_snr.pkl'), 'wb') as f:
            pickle.dump(acc_mod_snr, f)

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for snr in snrs:
        print(f"  SNR {snr:+3d} dB : {acc[snr]*100:6.2f}%")
    overall = np.mean(list(acc.values()))
    print(f"\n  Overall Average : {overall*100:.2f}%")
    print("=" * 50)

    return acc, acc_mod_snr
