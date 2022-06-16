import numpy as np
from numpy.typing import ArrayLike
import scipy as sp

#Time domain features
FEATURES_TIME = {
'mean_nni': lambda ppi:np.mean(ppi),
'sdnn' : lambda ppi:np.std(ppi, ddof=1),
'rmssd' : lambda ppi:np.sqrt(np.mean(np.diff(ppi) ** 2)),
'sdsd' : lambda ppi:np.std(np.diff(ppi)),
'nni_50' : lambda ppi:sum(np.abs(np.diff(ppi)) > 0.05),
'pnni_50' : lambda ppi:100 * sum(np.abs(np.diff(ppi)) > 0.05) / len(ppi),
'nni_20' : lambda ppi:sum(np.abs(np.diff(ppi)) > 0.02),
'pnni_20' : lambda ppi:100 * sum(np.abs(np.diff(ppi)) > 0.02) / len(ppi),
'cvnni' : lambda ppi:np.std(ppi, ddof=1) / np.mean(ppi),
'cvsd' : lambda ppi:np.sqrt(np.mean(np.diff(ppi) ** 2)) / np.mean(ppi),
'median_nni' : lambda ppi:np.median(ppi),
'range_nni' : lambda ppi:max(ppi) - min(ppi),
'mean_hr' : lambda ppi:np.mean(np.divide(60000, ppi)),
'min_hr' : lambda ppi:min(np.divide(60000, ppi)),
'max_hr' : lambda ppi:max(np.divide(60000, ppi)),
'std_hr' : lambda ppi:np.std(np.divide(60000, ppi)),
'mad_nni' : lambda ppi:np.median(np.abs(ppi-np.median(ppi))),
'mcv_nni' : lambda ppi:np.median(np.abs(ppi-np.median(ppi)))/np.median(ppi),
'iqr_nni': lambda ppi:sp.stats.iqr(ppi),
}


def hrv_time_features(ppi, prefix: str='hrv') -> dict:
    """Calculates time-domain features.

    MeanNN: The mean of the RR intervals.
    SDNN: the standard deviation of intervals. Often calculated over a 24-hour period. 
    (SDANN, the standard deviation of the average intervals calculated over short periods, usually 5 minutes)
    RMSSD: the square root of the mean of the squares of the successive differences between adjacent intervals
    SDSD: the standard deviation of the successive differences between adjacent intervals
    NN50: the number of pairs of successive intervals that differ by more than 50 ms
    pNN50: the proportion of NN50 divided by the total number of intervals
    NN20: the number of pairs of successive intervals that differ by more than 20 ms
    pNN20: the proportion of NN20 divided by the total number of intervals
    CVNN: The standard deviation of the RR intervals (SDNN) divided by the mean of the RR
        intervals (MeanNN).
    CVSD: The root mean square of the sum of successive differences (RMSSD) divided by the
        mean of the RR intervals (MeanNN). 
    MedianNN: The median of the absolute values of the successive differences between RR intervals.
    rangeNN: The range of the NN intervals
    meanHR: The mean HR
    maxHR: The maximum HR
    minHR: The minimum HR
    stdHR: The standard deviation of the HR
    MadNN: The median absolute deviation of the RR intervals.
    HCVNN: The median absolute deviation of the RR intervals (MadNN) divided by the median
        of the absolute differences of their successive differences (MedianNN).
    IQRNN: The interquartile range (IQR) of the RR intervals.    
    
        -**SDANN1**, **SDANN2**, **SDANN5**: The standard deviation of average RR intervals extracted from n-minute segments of
        time series data (1, 2 and 5 by default). Note that these indices require a minimal duration of signal to be computed
        (3, 6 and 15 minutes respectively) and will be silently skipped if the data provided is too short.
        -**SDNNI1**, **SDNNI2**, **SDNNI5**: The mean of the standard deviations of RR intervals extracted from n-minute
        segments of time series data (1, 2 and 5 by default). Note that these indices require a minimal duration of signal to
        be computed (3, 6 and 15 minutes respectively) and will be silently skipped if the data provided is too short.
        - **TINN**: A geometrical parameter of the HRV, or more specifically, the baseline width of
        the RR intervals distribution obtained by triangular interpolation, where the error of least
        squares determines the triangle. It is an approximation of the RR interval distribution.
        - **HTI**: The HRV triangular index, measuring the total number of RR intervals divded by the
        height of the RR intervals histogram.

    Args:
        sig (ArrayLike): Signal to be analyzed.
        fs (float): Sampling rate
        type (str, optional): Type of feature calculation, should be 'segment' or 'cycle'. Defaults to None.
        prefix (str, optional): Prefix for signal type. Defaults to 'signal'.

    Raises:
        ValueError: if Type is not 'cycle' or 'segment'.

    Returns:
        dict: Dictionary of calculated features.
    """

    features_time={}

    for key,func in FEATURES_TIME.items():
        features_time["_".join([prefix, key])]=func(ppi)


    return features_time







