import numpy as np
from scipy import stats

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


def get_stat_features(signal, prefix="signal"):

    s_features = {}

    for k, f in STAT_FEATURES.items():
        s_features["_".join([prefix, k])] = f(signal)

    return s_features
