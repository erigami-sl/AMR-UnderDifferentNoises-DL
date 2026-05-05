import numpy as np

def apply_fading(X, channel_type='rayleigh', K=1.0, block_fading=True):
    """
    Apply flat fading (Rayleigh or Rician) to a batch of IQ signals.
    
    Parameters
    ----------
    X : ndarray (N, 2, 128)
        Batch of input IQ signals (2 channels: I and Q).
    channel_type : str
        Type of fading: 'rayleigh' or 'rician'.
    K : float
        Rician K-factor (linear scale, not dB). Only used if channel_type='rician'.
    block_fading : bool
        If True, the fading coefficient is constant across the 128 samples of a single sequence.
        If False, the fading coefficient changes sample-by-sample (fast fading).
        
    Returns
    -------
    X_faded : ndarray (N, 2, 128)
        Signals with fading applied.
    """
    if X.ndim != 3 or X.shape[1] != 2:
        raise ValueError(f"Expected X shape (N, 2, 128), but got {X.shape}")
        
    N, channels, seq_len = X.shape
    
    # Convert to complex baseband representation
    I = X[:, 0, :]
    Q = X[:, 1, :]
    signal_complex = I + 1j * Q
    
    # Determine the shape of the fading coefficients
    h_shape = (N, 1) if block_fading else (N, seq_len)
    
    if channel_type.lower() == 'rayleigh':
        # Rayleigh fading coefficient: CN(0, 1)
        # Power E[|h|^2] = 1
        h = (np.random.randn(*h_shape) + 1j * np.random.randn(*h_shape)) / np.sqrt(2)
        
    elif channel_type.lower() == 'rician':
        # Rician fading coefficient
        # h = sqrt(K/(K+1)) * e^{j theta} + sqrt(1/(K+1)) * CN(0,1)
        # We assume the LOS component has theta = 0
        mu = np.sqrt(K / (K + 1))
        sigma = np.sqrt(1 / (2 * (K + 1)))
        
        h_los = mu + 0j
        h_nlos = sigma * (np.random.randn(*h_shape) + 1j * np.random.randn(*h_shape))
        h = h_los + h_nlos
        
    else:
        raise ValueError("Unsupported channel_type. Use 'rayleigh' or 'rician'.")
        
    # Apply the fading to the complex signal
    faded_signal_complex = signal_complex * h
    
    # Convert back to real-valued (I, Q) tensor
    X_faded = np.zeros_like(X)
    X_faded[:, 0, :] = np.real(faded_signal_complex)
    X_faded[:, 1, :] = np.imag(faded_signal_complex)
    
    return X_faded

def generate_faded_dataset(X_dict, channel_type='rayleigh', K=1.0):
    """
    Applies fading to the entire dataset dictionary and returns a new dictionary.
    
    Parameters
    ----------
    X_dict : dict
        Original dataset dict where keys are (mod_type, snr) and values are (N, 2, 128) arrays.
    channel_type : str
        Type of fading: 'rayleigh' or 'rician'.
    K : float
        Rician K-factor.
        
    Returns
    -------
    X_dict_faded : dict
        New dictionary with faded signals.
    """
    X_dict_faded = {}
    for key, X in X_dict.items():
        X_dict_faded[key] = apply_fading(X, channel_type=channel_type, K=K)
    return X_dict_faded
