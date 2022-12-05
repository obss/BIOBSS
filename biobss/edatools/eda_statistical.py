import numpy as np
from scipy import stats

# STAT_FEATURES : Dictionary of statistical features


STAT_FEATURES = {
    "mean": lambda x: np.mean(x),
    "std": lambda x: np.std(x),
    "max": lambda x: np.max(x),
    "min": lambda x: np.min(x),
    "range": lambda x: np.max(x) - np.min(x),
    "kurtosis": lambda x: stats.kurtosis(x),
    "skew": lambda x: stats.skew(x),
    "momentum": lambda x: stats.moment(x, 2),
}


def get_feature_names():

    return STAT_FEATURES.keys()


def eda_stat_features(signal: np.ndarray, prefix="signal") -> dict:
    """This methods takes an 1-D Array input and calculate statistical features over it.

    Features : 
    Mean, Standard Deviation, Max, Min, Range, Kurtosis, Skew, Momentum
    Momentum : The second moment of the signal.

    Args:
        signal (ArrayLike): input signal for statistical feature extraction
        prefix (str, optional): prefix for the signal name. Defaults to "signal".

    Returns:
        dict: Dictionary of statistical features from given signal
    """

    s_features = {}

    for k, f in STAT_FEATURES.items():
        s_features["_".join([prefix, k])] = f(signal)

    return s_features
