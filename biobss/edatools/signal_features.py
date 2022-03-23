import numpy as np
import neurokit2 as nk
from copy import copy

SIGNAL_FEATURES = {
    "rms": lambda x: rms(x),
    "al": lambda x: calculate_al(x),
    "in": lambda x: calculate_in(x),
    "ap": lambda x: calculate_ap(x),
}


def get_feature_names():
    return SIGNAL_FEATURES.keys()


def get_signal_features(signal, prefix="signal"):

    s_features = {}
    for k, f in SIGNAL_FEATURES.items():
        s_features["_".join([prefix, k])] = f(signal)

    return s_features


def rms(sig):

    rms_ = lambda x: x**2
    rms_func = np.vectorize(rms_)
    sig_ = rms_func(sig)
    tot = sig_.sum()
    tot = tot / len(sig)
    tot = np.sqrt(tot)
    return tot


def calculate_al(sig):
    sig = np.array(sig)
    sig1 = copy(sig[1:])
    sig2 = copy(sig[:-1])
    alsc_ = lambda x: np.sqrt(1 + x**2)
    alsc_func = np.vectorize(alsc_)
    tot = alsc_func(sig1 - sig2)
    tot = tot.sum()
    return tot


def calculate_in(sig):
    return np.abs(sig).sum()


def calculate_ap(sig):

    rms_ = lambda x: x**2
    rms_func = np.vectorize(rms_)
    sig_ = rms_func(sig)
    apsc = sig_.sum()
    apsc = apsc / len(sig)

    return apsc
