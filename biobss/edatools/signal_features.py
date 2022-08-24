import numpy as np
from copy import copy
from numpy.typing import ArrayLike

SIGNAL_FEATURES = {
    "rms": lambda x: calculate_rms(x),
    "acr_length": lambda x: calculate_arc_length(x),
    "integral": lambda x: calculate_integral(x),
    "average_power": lambda x: calculate_avg_pow(x),
}


def get_feature_names():
    return SIGNAL_FEATURES.keys()


def get_signal_features(signal: ArrayLike, prefix="signal") -> dict:
    """This method calculates features over a given signal.

    RMS (Root Mean Square):
    AL  (Arc Length)
    IN  (Integral)
    AP  (Normalized Average Power)

    Args:
        signal (ArrayLike): input signal
        prefix (str, optional): prefix for signal name. Defaults to "signal".

    Returns:
        dict: _description_
    """

    signal = np.array(signal)
    signal = signal.flatten()
    s_features = {}
    for k, f in SIGNAL_FEATURES.items():
        s_features["_".join([prefix, k])] = f(signal)

    return s_features


def calculate_rms(sig):

    def rms_(x): return x**2
    rms_func = np.vectorize(rms_)
    sig_ = rms_func(sig)
    tot = sig_.sum()
    tot = tot / len(sig)
    tot = np.sqrt(tot)
    return tot


def calculate_arc_length(sig):
    # This is the arc length of the signal
    sig = np.array(sig)
    sig1 = copy(sig[1:])
    sig2 = copy(sig[:-1])
    def alsc_(x): return np.sqrt(1 + x**2)
    alsc_func = np.vectorize(alsc_)
    tot = alsc_func(sig1 - sig2)
    tot = tot.sum()
    return tot


def calculate_integral(sig):
    # This is the integral of the signal
    return np.abs(sig).sum()


def calculate_avg_pow(sig):
    # This is the normalized average power of the signal

    def rms_(x): return x**2
    rms_func = np.vectorize(rms_)
    sig_ = rms_func(sig)
    apsc = sig_.sum()
    apsc = apsc / len(sig)

    return apsc
