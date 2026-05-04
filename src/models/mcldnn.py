"""
MCLDNN — Multi-Channel Long Short-Term Memory Deep Neural Network
=================================================================
A Spatiotemporal Multi-Channel Learning Framework for Automatic Modulation Recognition.

Reference:
    Xu, J., Luo, C., Parr, G., & Luo, Y. (2020).
    "A Spatiotemporal Multi-Channel Learning Framework for Automatic Modulation Recognition."
    IEEE Wireless Communications Letters, 9(10), 1629-1632.

Architecture:
    - 3 Inputs: IQ combined (2×128×1), I channel (128×1), Q channel (128×1)
    - Separate-channel Conv1D branches for I and Q
    - Combined Conv2D branch for IQ
    - Concatenation + Conv2D fusion
    - 2-layer LSTM for temporal features
    - Dense layers with SELU activation + Dropout
    - Softmax output (11 classes)

Migrated from TF1/Keras2 to TF2/tf.keras.
"""

import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, Conv2D, Dropout,
    concatenate, Flatten, Reshape, LSTM
)


def MCLDNN(weights=None,
           input_shape1=[2, 128],
           input_shape2=[128, 1],
           classes=11,
           **kwargs):
    """
    Build the MCLDNN model.

    Parameters
    ----------
    weights : str or None
        Path to pre-trained weights file. None for random initialization.
    input_shape1 : list
        Shape of the IQ combined input [2, 128].
    input_shape2 : list
        Shape of individual I/Q channel input [128, 1].
    classes : int
        Number of modulation classes (default: 11).

    Returns
    -------
    model : tf.keras.Model
        Compiled MCLDNN model.
    """
    if weights is not None and not os.path.exists(weights):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.5  # dropout rate

    # --- Inputs ---
    input1 = Input(input_shape1 + [1], name='input1')  # IQ combined: (2, 128, 1)
    input2 = Input(input_shape2, name='input2')         # I channel:   (128, 1)
    input3 = Input(input_shape2, name='input3')         # Q channel:   (128, 1)

    # --- Separate Channel Convolutional Branches ---
    # IQ combined branch
    x1 = Conv2D(50, (2, 8), padding='same', activation='relu',
                name='conv1_1', kernel_initializer='glorot_uniform')(input1)

    # I channel branch
    x2 = Conv1D(50, 8, padding='causal', activation='relu',
                name='conv1_2', kernel_initializer='glorot_uniform')(input2)
    x2_reshape = Reshape([-1, 128, 50], name='reshape_i')(x2)

    # Q channel branch
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
    # TF2: LSTM replaces CuDNNLSTM (auto-uses CuDNN when GPU available)
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

    # Load weights
    if weights is not None:
        model.load_weights(weights)

    return model


if __name__ == '__main__':
    model = MCLDNN(None, classes=11)
    model.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    )
    model.summary()