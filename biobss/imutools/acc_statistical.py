import numpy as np
from scipy import signal, stats

STAT_FEATURES = {
    "mean": lambda sig: np.mean(sig),
    "std": lambda sig: np.std(sig),
    "mad": lambda sig: np.mean(np.abs(sig - np.mean(sig))),
    "min": lambda sig: np.min(sig),
    "max": lambda sig: np.max(sig),
    "range": lambda sig: np.max(sig) - np.min(sig),
    "median": lambda sig: np.median(sig),
    "medad": lambda sig: np.median(np.abs(sig - np.median(sig))),
    "iqr": lambda sig: np.percentile(sig, 75) - np.percentile(sig, 25),
    "ncount": lambda sig: np.sum(sig < 0),
    "pcount": lambda sig: np.sum(sig > 0),
    "abmean": lambda sig: np.sum(sig > np.mean(sig)),
    "npeaks": lambda sig: len(signal.find_peaks(sig)[0]),
    "skew": lambda sig: stats.skew(sig),
    "kurtosis": lambda sig: stats.kurtosis(sig),
    "energy": lambda sig: np.sum(sig ** 2) / 100,
    "momentum": lambda sig: stats.moment(sig, 2),
}


def acc_stat_features(signals: list, signal_names: list, sampling_rate: float, magnitude: bool = False) -> dict:
    """Calculates statistical features.

    From https://towardsdatascience.com/feature-engineering-on-time-series-data-transforming-signal-data-of-a-smartphone-accelerometer-for-72cbe34b8a60

    mean: mean of the signal amplitude
    std: standard deviation of the signal amplitude
    mad: mean absolute deviation of the signal amplitude
    min: minimum value of the signal amplitude
    max: maximum value of the signal amplitude
    range: difference of maximum and minimum values of the signal amplitude
    median: median value of the signal amplitude
    medad: median absolute deviation of the signal amplitude
    iqr: interquartile range of the signal amplitude
    ncount: number of negative values
    pcount: number of positibe values
    abmean: number of values above mean
    npeaks: number of peaks
    skew: Skewness of the signal
    kurtosis: Kurtosis of the signal
    energy: signal energy (the mean of sum of squares of the values in a window)
    momentum: Signal momentum

    Args:
        signals (list): List of input signal(s).
        signal_names (list): List of signal name(s).
        sampling_rate (float): Sampling rate of the ACC signal(s) (Hz).
        magnitude (bool, optional): If True, features are also calculated for magnitude signal. Defaults to False.

    Returns:
        dict: Dictionary of statistical features.
    """

    if np.ndim(signals) == 1:
        signals = [signals]
    if isinstance(signal_names, str):
        signal_names = [signal_names]

    data = dict(zip(signal_names, signals))

    if magnitude:
        sum = 0
        for sig in signals:
            sum += np.square(sig)

        magn = np.sqrt(sum)
        data["magn"] = magn

    features_stat = {}
    for signal_name, signal in data.items():
        for key, func in STAT_FEATURES.items():
            try:
                features_stat["_".join([signal_name, key])] = func(signal)
            except:
                features_stat["_".join([signal_name, key])] = np.nan

    return features_stat
