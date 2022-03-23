import numpy as np
import neurokit2 as nk


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
    tot = 0
    for s in sig:
        tot += np.abs(s) * np.abs(s)
    tot = tot / len(sig)
    tot = np.sqrt(tot)
    return tot


def calculate_al(sig):

    alsc = 0
    for i in range(1, len(sig)):
        alsc += np.sqrt(1 + (sig[i] - sig[i - 1]) ** 2)
    return alsc


def calculate_in(sig):
    return np.abs(sig).sum()


def calculate_ap(sig):
    apsc = 0
    for i in range(0, len(sig)):
        apsc += sig[i] ** 2

    return apsc
