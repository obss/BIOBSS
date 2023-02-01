from copy import copy

import numpy as np
from numpy.typing import ArrayLike

SIGNAL_FEATURES = {
    "rms": lambda x: _calculate_rms(x),
    "acr_length": lambda x: _calculate_arc_length(x),
    "integral": lambda x: _calculate_integral(x),
    "average_power": lambda x: _calculate_avg_pow(x),
}


def get_feature_names():
    return SIGNAL_FEATURES.keys()


def eda_signal_features(signal: ArrayLike, prefix="signal") -> dict:
    """Calculates EDA features.

    rms : Root mean square of the signal
    acr_length : Arc length of the signal
    integral : Integral of the signal
    average_power: Normalized average power of the signal

    Args:
        signal (ArrayLike): EDA signal.
        prefix (str, optional): Prefix for the feature. Defaults to "eda".

    Returns:
        dict: Dictionary of calculated features.
    """
    signal = np.array(signal)
    signal = signal.flatten()
    s_features = {}
    for k, f in SIGNAL_FEATURES.items():
        try:
            s_features["_".join([prefix, k])] = f(signal)
        except:
            s_features["_".join([prefix, k])] = np.nan

    return s_features


def _calculate_rms(sig: ArrayLike) -> float:
    def rms_(x):
        return x ** 2

    rms_func = np.vectorize(rms_)
    sig_ = rms_func(sig)
    tot = sig_.sum()
    tot = tot / len(sig)
    tot = np.sqrt(tot)

    return tot


def _calculate_arc_length(sig: ArrayLike) -> float:

    # This is the arc length of the signal
    sig = np.array(sig)
    sig1 = copy(sig[1:])
    sig2 = copy(sig[:-1])

    def alsc_(x):
        return np.sqrt(1 + x ** 2)

    alsc_func = np.vectorize(alsc_)
    tot = alsc_func(sig1 - sig2)
    tot = tot.sum()

    return tot


def _calculate_integral(sig: ArrayLike) -> float:
    # This is the integral of the signal
    return np.abs(sig).sum()


def _calculate_avg_pow(sig: ArrayLike) -> float:
    # This is the normalized average power of the signal

    def rms_(x):
        return x ** 2

    rms_func = np.vectorize(rms_)
    sig_ = rms_func(sig)
    apsc = sig_.sum()
    apsc = apsc / len(sig)

    return apsc
