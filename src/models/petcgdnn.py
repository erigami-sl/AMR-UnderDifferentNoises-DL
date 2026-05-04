"""
PET-CGDNN — Parameter Estimation & Transformation + CNN-GRU Deep Neural Network
================================================================================
An Efficient Deep Learning Model for Automatic Modulation Recognition
Based on Parameter Estimation and Transformation.

Reference:
    Zhang, F., Luo, C., Xu, J., & Luo, Y. (2021).
    "An Efficient Deep Learning Model for Automatic Modulation Recognition
     Based on Parameter Estimation and Transformation."
    IEEE Communications Letters, 25(10), 3287-3291.

Architecture:
    - 3 Inputs: IQ combined (128×2×1), I channel (128,), Q channel (128,)
    - Learnable phase estimation: θ = Dense(1)(Flatten(IQ))
    - Phase rotation: y1 = I·cos(θ) + Q·sin(θ), y2 = Q·cos(θ) − I·sin(θ)
    - 2-layer Conv2D for spatial features
    - GRU for temporal features
    - Softmax output (11 classes)

Migrated from TF1/Keras2 to TF2/tf.keras.
"""

import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, Flatten, Reshape,
    Activation, Lambda, Multiply, Add, Subtract,
    concatenate, GRU
)


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

    Parameters
    ----------
    weights : str or None
        Path to pre-trained weights file. None for random initialization.
    input_shape : list
        Shape of the IQ combined input [128, 2].
    input_shape2 : list
        Shape of individual I/Q channel input [128].
    classes : int
        Number of modulation classes (default: 11).

    Returns
    -------
    model : tf.keras.Model
        Compiled PET-CGDNN model.
    """
    if weights is not None and not os.path.exists(weights):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.5  # dropout rate

    # --- Inputs ---
    inp = Input(input_shape + [1], name='input1')    # IQ combined: (128, 2, 1)
    input1 = Input(input_shape2, name='input2')      # I channel:   (128,)
    input2 = Input(input_shape2, name='input3')      # Q channel:   (128,)

    # --- Parameter Estimation (Phase θ) ---
    x1 = Flatten(name='flatten_iq')(inp)
    x1 = Dense(1, name='phase_estimate')(x1)
    x1 = Activation('linear', name='phase_linear')(x1)

    # --- Phase Transformation ---
    cos1 = Lambda(_cos, name='lambda_cos')(x1)
    sin1 = Lambda(_sin, name='lambda_sin')(x1)

    # y1 = I·cos(θ) + Q·sin(θ)
    x11 = Multiply(name='mul_i_cos')([input1, cos1])
    x12 = Multiply(name='mul_q_sin')([input2, sin1])
    y1 = Add(name='add_y1')([x11, x12])

    # y2 = Q·cos(θ) − I·sin(θ)
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
    # TF2: GRU replaces CuDNNGRU (auto-uses CuDNN when GPU available)
    x4 = Reshape(target_shape=(117, 25), name='reshape4')(x3)
    x4 = GRU(units=128, name='gru')(x4)

    # --- Classifier ---
    x = Dense(classes, activation='softmax', name='softmax')(x4)

    model = Model(inputs=[inp, input1, input2], outputs=x)

    # Load weights
    if weights is not None:
        model.load_weights(weights)

    return model


if __name__ == '__main__':
    model = PETCGDNN(None, classes=11)
    model.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    )
    model.summary()