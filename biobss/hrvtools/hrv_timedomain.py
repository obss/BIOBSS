import numpy as np
import scipy as sp
from numpy.typing import ArrayLike

# Time domain features
FEATURES_TIME = {
    "mean_nni": lambda ppi: np.mean(ppi),
    "sdnn": lambda ppi: np.std(ppi, ddof=1),
    "rmssd": lambda ppi: np.sqrt(np.mean(np.diff(ppi) ** 2)),
    "sdsd": lambda ppi: np.nanstd(np.diff(ppi), ddof=1),
    "nni_50": lambda ppi: sum(np.abs(np.diff(ppi)) > 50),
    "pnni_50": lambda ppi: 100 * sum(np.abs(np.diff(ppi)) > 50) / len(ppi),
    "nni_20": lambda ppi: sum(np.abs(np.diff(ppi)) > 20),
    "pnni_20": lambda ppi: 100 * sum(np.abs(np.diff(ppi)) > 20) / len(ppi),
    "cvnni": lambda ppi: np.std(ppi, ddof=1) / np.mean(ppi),
    "cvsd": lambda ppi: np.sqrt(np.mean(np.diff(ppi) ** 2)) / np.mean(ppi),
    "median_nni": lambda ppi: np.median(ppi),
    "range_nni": lambda ppi: max(ppi) - min(ppi),
    "mean_hr": lambda ppi: np.mean(np.divide(60000, ppi)),
    "min_hr": lambda ppi: min(np.divide(60000, ppi)),
    "max_hr": lambda ppi: max(np.divide(60000, ppi)),
    "std_hr": lambda ppi: np.std(np.divide(60000, ppi)),
    "mad_nni": lambda ppi: np.nanmedian(np.abs(ppi - np.nanmedian(ppi))),
    "mcv_nni": lambda ppi: np.median(np.abs(ppi - np.median(ppi))) / np.median(ppi),
    "iqr_nni": lambda ppi: sp.stats.iqr(ppi),
}


def hrv_time_features(ppi: ArrayLike, sampling_rate: int, prefix: str = "hrv") -> dict:
    """Calculates time-domain hrv parameters.

    mean_nni: mean of peak to peak intervals
    sdnn: standard deviation of peak to peak intervals. Often calculated over a 24-hour period.
    rmssd: root mean square of successive differences between peak to peak intervals
    sdsd: standard deviation of successive differences between peak to peak intervals
    nni_50: number of pairs of successive intervals that differ by more than 50 ms
    pnni_50: ratio of nni_50 to total number of intervals
    nni_20: number of pairs of successive intervals that differ by more than 20 ms
    pnni_20: ratio of nni_20 to total number of intervals
    cvnni: ratio of sdnn to mean_nni
    cvsd: ratio of rmssd to mean_nni
    median_nni: median of absolute values of successive differences between peak to peak intervals
    range_nni: range of peak to peak intervals
    mean_hr: mean heart rate
    min_hr: minimum heart rate
    max_hr: maximum heart rate
    std_hr: standard deviation of heart rate
    mad_nni: mean absolute deviation of peak to peak intervals
    mcv_nni: ratio of mead_nni to median_nni
    iqr_nni: interquartile range of peak to peak intervals

    Args:
        ppi (ArrayLike): Peak-to-peak interval array (miliseconds).
        prefix (str, optional): Prefix for the calculated parameters. Defaults to 'hrv'.

    Returns:
        dict: Dictionary of time-domain hrv parameters.
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    features_time = {}
    for key, func in FEATURES_TIME.items():
        try:
            features_time["_".join([prefix, key])] = func(ppi)
        except:
            features_time["_".join([prefix, key])] = np.nan

    return features_time
