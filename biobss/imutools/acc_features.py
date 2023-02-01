from typing import Callable

import numpy as np

from biobss.imutools.acc_correlation import *
from biobss.imutools.acc_freqdomain import *
from biobss.imutools.acc_statistical import *


def get_domain_function(domain: str) -> Callable:

    if domain == "Corr":
        return acc_corr_features
    elif domain == "Freq":
        return acc_freq_features
    elif domain == "Stat":
        return acc_stat_features
    else:
        raise ValueError("Unknown domain:", domain)


def get_acc_features(
    signals: list,
    signal_names: list,
    sampling_rate: float,
    feature_types: list = ["Freq", "Stat", "Corr"],
    magnitude: bool = False,
) -> dict:
    """Calculates ACC features.

    Args:
        signals (list): List of ACC signals for different axes.
        signal_names (list): List of ACC signal names for different axes. It must have the same order with the signals.
        sampling_rate (float): Sampling rate of the ACC signals (Hz).
        feature_types (list, optional): Feature types to be calculated. It can be a list of 'Freq', 'Stat' and 'Corr'. Defaults to ['Freq','Stat','Corr'].
        magnitude (bool, optional): If True, features are also calculated for magnitude signal. Defaults to False.

    Raises:
        ValueError: If sampling rate is not greater than zero.
        ValueError: If feature type is invalid.

    Returns:
        dict: Dictionary of ACC features.
    """
    if np.ndim(signals) == 1:
        signals = [signals]
    if isinstance(signal_names, str):
        signal_names = [signal_names]

    data = dict(zip(signal_names, signals))

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    feature_types = [x.capitalize() for x in feature_types]
    valid_types = ["Freq", "Stat", "Corr"]

    features = {}
    for domain in feature_types:
        if domain not in valid_types:
            raise ValueError("invalid feature type: " + domain)
        else:
            domain_function = get_domain_function(domain)
            features.update(domain_function(signals=signals, signal_names=signal_names, sampling_rate=sampling_rate))

            if domain in ["Freq", "Stat"]:
                if magnitude:
                    sum = 0
                    for sig in data.values():
                        sum += np.square(sig)
                    magn = np.sqrt(sum)
                    features.update(domain_function(signals=[magn], signal_names=["magn"], sampling_rate=sampling_rate))

    return features
