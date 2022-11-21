import numpy as np
from numpy.typing import ArrayLike
import scipy as sp

#Time domain features
FEATURES_TIME = {
'mean_nni': lambda ppi:np.mean(ppi),
'sdnn' : lambda ppi:np.std(ppi, ddof=1),
'rmssd' : lambda ppi:np.sqrt(np.mean(np.diff(ppi) ** 2)),
'sdsd' : lambda ppi:np.nanstd(np.diff(ppi),ddof=1),
'nni_50' : lambda ppi:sum(np.abs(np.diff(ppi)) > 50),
'pnni_50' : lambda ppi:100 * sum(np.abs(np.diff(ppi)) > 50) / len(ppi),
'nni_20' : lambda ppi:sum(np.abs(np.diff(ppi)) > 20),
'pnni_20' : lambda ppi:100 * sum(np.abs(np.diff(ppi)) > 20) / len(ppi),
'cvnni' : lambda ppi:np.std(ppi, ddof=1) / np.mean(ppi),
'cvsd' : lambda ppi:np.sqrt(np.mean(np.diff(ppi) ** 2)) / np.mean(ppi),
'median_nni' : lambda ppi:np.median(ppi),
'range_nni' : lambda ppi:max(ppi) - min(ppi),
'mean_hr' : lambda ppi:np.mean(np.divide(60000, ppi)),
'min_hr' : lambda ppi:min(np.divide(60000, ppi)),
'max_hr' : lambda ppi:max(np.divide(60000, ppi)),
'std_hr' : lambda ppi:np.std(np.divide(60000, ppi)),
'mad_nni' : lambda ppi:np.nanmedian(np.abs(ppi-np.nanmedian(ppi))),
'mcv_nni' : lambda ppi:np.median(np.abs(ppi-np.median(ppi)))/np.median(ppi),
'iqr_nni': lambda ppi:sp.stats.iqr(ppi),
}

def hrv_time_features(ppi: ArrayLike, sampling_rate:int, prefix: str='hrv') -> dict:
    """Calculates time-domain hrv parameters.

    mean_nni: The mean of the RR intervals.
    sdnn: the standard deviation of intervals. Often calculated over a 24-hour period. 
    (sdann, the standard deviation of the average intervals calculated over short periods, usually 5 minutes)
    rmssd: the square root of the mean of the squares of the successive differences between adjacent intervals
    sdsd: the standard deviation of the successive differences between adjacent intervals
    nni_50: the number of pairs of successive intervals that differ by more than 50 ms
    pnni_50: the proportion of NN50 divided by the total number of intervals
    nni_20: the number of pairs of successive intervals that differ by more than 20 ms
    pnni_20: the proportion of NN20 divided by the total number of intervals
    cvnni: The standard deviation of the RR intervals (sdnn) divided by the mean of the RR
        intervals (mean_nni).
    cvsd: the square root of the mean of the squares of the successive differences between adjacent intervals (rmssd) divided by the
        mean of the RR intervals (mean_nni). 
    median_nni: The median of the absolute values of the successive differences between RR intervals.
    range_nni: The range of the NN intervals
    mean_hr: The mean HR
    min_hr: The minimum HR
    max_hr: The maximum HR
    std_hr: The standard deviation of the HR
    mad_nni: The median absolute deviation of the RR intervals.
    mcv_nni: The median absolute deviation of the RR intervals (mad_nni) divided by the median
        of the absolute differences of their successive differences (median_nni).
    iqr_nni: The interquartile range (IQR) of the RR intervals.    

    Args:
        ppi (ArrayLike): Peak-to-peak interval array (miliseconds).
        prefix (str, optional): Prefix for the calculated parameters. Defaults to 'hrv'.

    Returns:
        dict: Dictionary of time-domain hrv parameters.
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")
        
    features_time={}
    for key,func in FEATURES_TIME.items():
        features_time["_".join([prefix, key])]=func(ppi)

    return features_time







