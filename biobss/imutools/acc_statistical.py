import numpy as np
from scipy import stats, signal
from numpy.typing import ArrayLike

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
    "energy": lambda sig: np.sum(sig**2)/100,
    "momentum": lambda sig: stats.moment(sig, 2),
}


def get_stat_features(sig: ArrayLike, sampling_rate, prefix) -> dict:
    """Calculates statistical features.

    From https://towardsdatascience.com/feature-engineering-on-time-series-data-transforming-signal-data-of-a-smartphone-accelerometer-for-72cbe34b8a60

    mean: Mean value of the signal.
    std: Standard deviation of the signal.
    mad: Mean absolute deviation of the signal.
    min: Minimum value of the signal.
    max: Maximum value of the signal.
    range: Difference of maximum and minimum values.
    median: Median value of the signal.
    medad: Median absolute deviation of the signal.
    iqr: Interquartile range of the signal. 
    ncount: Number of negative values.
    pcount: Number of positive values
    abmean: Number of values above mean 
    npeaks: Number of peaks
    skew: Skewness
    kurtosis: Kurtosis
    energy: Signal energy
    momentum: Signal momentum
    average resultant acceleration: [i.mean() for i in ((pd.Series(x_list)**2 + pd.Series(y_list)**2 + pd.Series(z_list)**2)**0.5)]
    signal magnitude area: pd.Series(x_list).apply(lambda x: np.sum(abs(x)/100)) + pd.Series(y_list).apply(lambda x: np.sum(abs(x)/100)) + pd.Series(z_list).apply(lambda x: np.sum(abs(x)/100)) 

    Args:
        sig (ArrayLike): Input signal
        sampling_rate (_type_): Sampling rate
        prefix (_type_): Prefix 

    Returns:
        dict: Dictionary of statistical features
    """

    features_stat={}

    for key,func in STAT_FEATURES.items():
        features_stat["_".join([prefix, key])]=func(sig)


    return features_stat
