import numpy as np
from numpy.typing import ArrayLike


def normalize_signal(signal: ArrayLike, method: str = "zscore") -> ArrayLike:
    """Normalizes a signal.

    Args:
        signal (ArrayLike): Input signal.
        method (str, optional): Normalization method. Defaults to 'zscore'.

    Raises:
        ValueError: If method is not 'zscore' or 'minmax'.

    Returns:
        ArrayLike: Normalized signal
    """
    # Need to add signal check
    epsilon = 1e-100
    if method == "zscore":
        return (signal - np.mean(signal)) / (np.std(signal) + epsilon)
    elif method == "minmax":
        return (signal - signal.min()) / (signal.max() - signal.min() + epsilon)
    else:
        raise ValueError(f"Unknown method '{method}', available values are [zscore, minmax].")
