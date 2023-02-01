import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

# STAT_FEATURES : Dictionary of statistical features
STAT_FEATURES = {
    "mean": lambda sig: np.mean(sig),
    "std": lambda sig: np.std(sig),
    "max": lambda sig: np.max(sig),
    "min": lambda sig: np.min(sig),
    "range": lambda sig: np.max(sig) - np.min(sig),
    "kurtosis": lambda sig: stats.kurtosis(sig),
    "skew": lambda sig: stats.skew(sig),
    "momentum": lambda sig: stats.moment(sig, 2),
}


def get_feature_names():
    return STAT_FEATURES.keys()


def eda_stat_features(signal: ArrayLike, prefix: str = "signal") -> dict:
    """Calculates statistical EDA features.

    mean: Mean of the signal
    std: Standard deviation of the signal
    max: Maaximum value of the signal
    min: Minimum value of the signal
    range: Range of the signal
    kurtosis: Kurtosis of the signal
    skew: Skewness of the signal
    momentum: The second moment of the signal

    Args:
        signal (ArrayLike): EDA signal.
        prefix (str, optional): Prefix for the feature. Defaults to "eda".

    Returns:
        dict: Dictionary of calculated features.
    """

    s_features = {}

    for k, f in STAT_FEATURES.items():
        try:
            s_features["_".join([prefix, k])] = f(signal)
        except:
            s_features["_".join([prefix, k])] = np.nan

    return s_features
