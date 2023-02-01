import itertools

import numpy as np
from scipy.stats import pearsonr


def acc_corr_features(signals: list, signal_names: list, sampling_rate: float) -> dict:
    """Calculates correlation features for ACC signals.
        For example:
        accx_accy_corr: correlation coefficient for x and y axes
        accx_accz_corr: correlation coefficient for x and z axes
        accy_accz_corr: correlation coefficient for y and z axes
    Args:
        signals (list): List of input signals.
        signal_names (list): List of signal names. It must have the same order with signal array.
        sampling_rate (float): Sampling rate of the ACC signal(s) (Hz).

    Returns:
        dict: Dictionary of correlation features.
    """
    if np.ndim(signals) == 1:
        signals = [signals]
    if isinstance(signal_names, str):
        signal_names = [signal_names]

    data = dict(zip(signal_names, signals))
    comb = list(itertools.combinations(signal_names, 2))

    corr_list = {}
    for i in comb:
        try:
            corr_list["_".join(i) + "_corr"] = pearsonr(data[i[0]], data[i[1]])[0]
        except:
            corr_list["_".join(i) + "_corr"] = np.nan

    return corr_list
