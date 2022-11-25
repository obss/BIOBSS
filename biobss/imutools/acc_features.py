import numpy as np
from numpy.typing import ArrayLike
from typing import Callable

from biobss.imutools.acc_freqdomain import *
from biobss.imutools.acc_statistical import *
from biobss.imutools.acc_correlation import *

def get_domain_function(domain:str) -> Callable:

    if domain == "Corr":
        return get_corr_features
    elif domain == "Freq":
        return get_freq_features
    elif domain == "Stat":
        return get_stat_features
    else:
        raise ValueError("Unknown domain:", domain)   

    
def get_acc_features(signals: ArrayLike, signal_names: ArrayLike, sampling_rate: float, prefix: str="acc", feature_types: ArrayLike=['Freq','Stat','Corr'], magnitude: bool=False) -> dict:
    """Calculates ACC features and returns a dictionary.

    Args:
        signals (dict): Dictionary of ACC signals for different axes.
        sampling_rate (float): Sampling rate of the ACC signals (Hz).
        prefix (str, optional): Prefix. Defaults to "acc".
        feature_types (ArrayLike, optional): Feature types to be calculated. It can be a list of 'Freq', 'Stat' and 'Corr'. Defaults to ['Freq','Stat','Corr'].
        magnitude (bool, optional): If True, features are also calculated for magnitude signal. Defaults to False.

    Raises:
        ValueError: If sampling rate is not greater than zero.
        ValueError: If feature type is invalid.

    Returns:
        dict: Dictionary of calculated features.
    """
    data = dict(zip(signal_names, signals))

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")
    
    feature_types = [x.capitalize() for x in feature_types]
    valid_types=['Freq','Stat','Corr']

    features={}
    
    for domain in feature_types:
        if domain not in valid_types:
            raise ValueError("invalid feature type: " + domain)
        else:
            if domain in ['Freq', 'Stat']:

                domain_function = get_domain_function(domain)
                features.update(domain_function(signals=signals,signal_names=signal_names,sampling_rate=sampling_rate))

                if magnitude:
                    sum=0
                    for sig in data.values():
                        sum += np.square(sig)
                    magn=np.sqrt(sum)
                    features.update(domain_function(signals=[magn],signal_names=['magn'],sampling_rate=sampling_rate))                
            else: 
                domain_function = get_domain_function(domain)
                features.update(domain_function(signals=signals,signal_names=signal_names,sampling_rate=sampling_rate))

    return features












