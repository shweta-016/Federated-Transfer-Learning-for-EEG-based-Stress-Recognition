# normalization.py
# Z-score normalization for EEG signals

import numpy as np


def z_score_normalize(signal):
    """
    Apply Z-score normalization.
    X_norm = (X - mean) / std
    """
    mean = np.mean(signal)
    std = np.std(signal) + 1e-6
    return (signal - mean) / std