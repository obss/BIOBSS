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

    
def get_acc_features(signals: dict, sampling_rate: float, prefix: str="acc", feature_types: ArrayLike=['Freq','Stat','Corr'], magnitude: bool=False) -> dict:
  
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
                for signal_name, signal in signals.items():
                    features.update(domain_function(signal,sampling_rate,prefix=signal_name))
                if magnitude:
                    sum=0
                    for sig in signals.values():
                        sum += np.square(sig)
                    magn=np.sqrt(sum)
                    features.update(domain_function(magn,sampling_rate,prefix="magn"))
            else: 
                domain_function = get_domain_function(domain)
                features.update(domain_function(signals, prefix=prefix))

    return features












