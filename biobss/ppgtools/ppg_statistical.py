import collections

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from biobss.common.signal_entropy import *

# Statistical features
FEATURES_STAT_CYCLE = {
    "mean_peaks": lambda _0, peaks_amp, _1, _2, _3: np.mean(peaks_amp),
    "std_peaks": lambda _0, peaks_amp, _1, _2, _3: np.std(peaks_amp),
}

FEATURES_STAT_SEGMENT = {
    "mean": np.mean,
    "median": np.median,
    "std": np.std,
    "pct_25": lambda sig: np.percentile(sig, 25),
    "pct_75": lambda sig: np.percentile(sig, 75),
    "mad": lambda sig: np.sum(sig - np.mean(sig)) / len(sig),
    "skewness": stats.skew,
    "kurtosis": stats.kurtosis,
    "entropy": lambda sig: calculate_shannon_entropy(sig),
}


def ppg_stat_features(
    sig: ArrayLike, sampling_rate: float, input_types: list, fiducials: dict = None, prefix: str = "ppg", **kwargs
) -> dict:
    """Calculates statistical features.

    Cycle-based features:
        mean_peaks: Mean of the peak amplitudes
        std_peaks: Standard deviation of the peak amplitudes

    Segment-based features:
        mean: Mean value of the signal
        median: Median value of the signal
        std: Standard deviation of the signal
        pct_25: 25th percentile of the signal
        pct_75 75th percentile of the signal
        mad: Mean absolute deviation of the signal
        skewness: Skewness of the signal
        kurtosis: Kurtosis of the signal
        entropy: Entropy of the signal

    Args:
        sig (ArrayLike): Signal to be analyzed.
        sampling_rate (float): Sampling rate of the signal (Hz).
        input_types (list): Type of feature calculation, should be 'segment' or 'cycle'.
        fiducials (dict, optional): Dictionary of fiducial point locations. Defaults to None.
        prefix (str, optional): Prefix for signal type. Defaults to 'ppg'.

    Kwargs:
        peaks_locs (ArrayLike): Array of peak locations
        troughs_locs (ArrayLike): Array of trough locations

    Raises:
        ValueError: If sampling rate is not greater than 0.
        ValueError: If 'peaks_locs' and/or 'troughs_locs' is not provided for the input_type 'cycle'.
        ValueError: If type is not 'cycle' or 'segment'.

    Returns:
        dict: Dictionary of calculated features.
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    input_types = [x.lower() for x in input_types]

    features_stat = {}
    for type in input_types:

        if type == "cycle":
            for key, func in FEATURES_STAT_CYCLE.items():
                if all(k in kwargs.keys() for k in ("peaks_locs", "troughs_locs")):
                    peaks_amp = sig[kwargs["peaks_locs"]]
                    try:
                        features_stat["_".join([prefix, key])] = func(
                            sig, peaks_amp, kwargs["peaks_locs"], kwargs["troughs_locs"], sampling_rate
                        )
                    except:
                        features_stat["_".join([prefix, key])] = np.nan
                else:
                    raise ValueError("Missing keyword arguments for the input_type: 'cycle'!")

        elif type == "segment":
            for key, func in FEATURES_STAT_SEGMENT.items():
                try:
                    features_stat["_".join([prefix, key])] = func(sig)
                except:
                    features_stat["_".join([prefix, key])] = np.nan

        else:
            raise ValueError("Type should be 'cycle' or 'segment'.")

    return features_stat
