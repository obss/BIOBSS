import numpy as np
from scipy import stats
import collections
from numpy.typing import ArrayLike

#Statistical features
FEATURES_STAT_CYCLE = {
'std_peaks': lambda _0,_1,peaks_locs,_2,_3: np.std(peaks_locs),
}

FEATURES_STAT_SEGMENT = {
'mean': np.mean,
'median': np.median,
'std': np.std,
'pct_25': lambda sig: np.percentile(sig, 25),
'pct_75': lambda sig: np.percentile(sig, 75),
'mad': lambda sig: np.sum(sig-np.mean(sig))/len(sig),
'skewness': stats.skew,
'kurtosis': stats.kurtosis,
'entropy': lambda sig:calculate_shannon_entropy(sig),
}

def get_stat_features(sig: ArrayLike,fs: float,type: str, prefix: str='signal', **kwargs) -> dict:
    """Calculates statistical features.

    Cycle-based features: 
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
        fs (float): Sampling rate
        type (str): Type of feature calculation, should be 'segment' or 'cycle'. Defaults to None.
        prefix (str, optional): Prefix for signal type. Defaults to 'signal'.

    Raises:
        ValueError: if type is not equal to 'cycle' or 'segment'.

    Returns:
        dict: Dictionary of calculated features.
    """

    if type=='cycle':
        features_stat={}
        for key,func in FEATURES_STAT_CYCLE.items():
            features_stat["_".join([prefix, key])]=func(sig,kwargs['peaks_amp'],kwargs['peaks_locs'],kwargs['troughs_locs'],fs)
    
    elif type=='segment':
        features_stat={}
        for key,func in FEATURES_STAT_SEGMENT.items():           
            features_stat["_".join([prefix, key])]=func(sig)

    else: 
        raise ValueError("Type should be 'cycle' or 'segment'.")

    return features_stat


def calculate_shannon_entropy(sig: ArrayLike) -> float:
    """Calculates shannon entropy of the signal.

    Args:
        sig (ArrayLike): Signal to be analyzed.

    Returns:
        float: Shannon entropy of the signal.
    """

    bases = collections.Counter([tmp_base for tmp_base in sig])
    # define distribution
    dist = [x/sum(bases.values()) for x in bases.values()]
 
    # use scipy to calculate entropy
    entropy_value = stats.entropy(dist, base=2)
 
    return entropy_value