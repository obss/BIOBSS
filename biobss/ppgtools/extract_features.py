from numpy.typing import ArrayLike
from typing import Tuple

from biobss.ppgtools.freqdomain_features import get_freq_features
from biobss.ppgtools.stat_features import get_stat_features
from biobss.ppgtools.timedomain_features import get_time_features

DOMAIN_FUNCTIONS = {
    'Time' : get_time_features,
    'Freq' : get_freq_features,   
    'Stat' : get_stat_features,
    }


def from_cycles(sig: ArrayLike, peaks_locs: ArrayLike, peaks_amp: ArrayLike, troughs_locs: ArrayLike ,troughs_amp: ArrayLike ,sampling_rate: float, feature_types: ArrayLike=['Time','Stat'], prefix: str='signal') -> dict:
    """Calculates cycle-based features.

    Args:
        sig (ArrayLike): Signal segment to be analyzed
        peaks_locs (ArrayLike): Peak locations
        peaks_amp (ArrayLike): Peak amplitudes
        troughs_locs (ArrayLike): Trough locations
        troughs_amp (ArrayLike): Trough amplitudes
        sampling_rate (float): Sampling rate
        feature_types (ArrayLike, optional): Types of features to be calculated. Defaults to ['Time','Stat'].
        prefix (str, optional): Prefix for signal type. Defaults to 'signal'.

    Raises:
        ValueError: if elements of feature_types are not equal to 'Time' or 'Stat'.

    Returns:
        dict: Dictionary of calculated features.
    """

    valid_types=['Time','Stat'] 
    features={}
    for domain in feature_types:
        if domain not in valid_types:
            raise ValueError("Invalid feature type: " + domain)
        else:
            features.update(DOMAIN_FUNCTIONS[domain](sig,sampling_rate,type='cycle',prefix=prefix, peaks_locs=peaks_locs, peaks_amp=peaks_amp, troughs_locs=troughs_locs, troughs_amp=troughs_amp))

    return features


def from_segment(sig: ArrayLike,sampling_rate: float, feature_types: ArrayLike=['Stat','Freq','Time'], prefix: str='signal') -> dict:
    """Calculates segment-based features.

    Args:
        sig (ArrayLike): Signal segment to be analyzed
        sampling_rate (float): Sampling rate
        feature_types (ArrayLike, optional): Types of features to be calculated. Defaults to ['Stat','Freq','Time'].
        prefix (str, optional): Prefix for signal type. Defaults to 'signal'.

    Raises:
        ValueError: if elements of feature_types are not equal to 'Time', 'Stat' or 'Freq'.

    Returns:
        dict: Dictionary of calculated features.
    """

    valid_types=['Time','Stat','Freq']   
    features={}
    for domain in feature_types:
        if domain not in valid_types:
            raise ValueError("invalid feature type: " + domain)
        else:
            features.update(DOMAIN_FUNCTIONS[domain](sig,sampling_rate,type='segment',prefix=prefix))

    return features












