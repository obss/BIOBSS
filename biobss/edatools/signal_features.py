import numpy as np
import neurokit2 as nk


def get_signal_features(signal, prefix="signal"):

    s_features = {}
    s_features[prefix + "_rmse"] = rmse(signal)  # root mean square
    s_features[prefix + "_al"] = calculate_alsc(signal)  # average arc length
    s_features[prefix + "_in"] = calculate_insc(signal)  # integral
    s_features[prefix + "_ap"] = calculate_apsc(signal)  # average power
    return s_features

def rmse(sig):
    tot = 0
    for s in sig:
        tot += np.abs(s) * np.abs(s)
    tot = tot / len(sig)
    tot = np.sqrt(tot)
    return tot


def calculate_alsc(sig):

    alsc = 0
    for i in range(1, len(sig)):
        alsc += np.sqrt(1 + (sig[i] - sig[i - 1]) ** 2)
    return alsc


def calculate_insc(sig):
    return np.abs(sig).sum()


def calculate_apsc(sig):
    apsc = 0
    for i in range(0, len(sig)):
        apsc += sig[i] ** 2

    return apsc
