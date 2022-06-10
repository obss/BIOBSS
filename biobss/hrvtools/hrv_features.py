from numpy.typing import ArrayLike
from typing import Callable
import numpy as np

from biobss.hrvtools.hrv_fd import get_freq_features
from biobss.hrvtools.hrv_td import get_time_features
from biobss.hrvtools.hrv_nl import get_nl_features

def get_domain_function(domain:str) -> Callable:

    if domain == "Time":
        return get_time_features
    elif domain == "Freq":
        return get_freq_features
    elif domain == "Nonlinear":
        return get_nl_features
    else:
        raise ValueError("Unknown domain:", domain)  


def get_hrv_features(sig: ArrayLike, sampling_rate: float, peaks_locs=None, troughs_locs=None, method='ppi', ppi=None, feature_types: ArrayLike=['Freq','Time','Nonlinear'], prefix: str='hrv') -> dict:

    valid_types=['Time','Freq','Nonlinear']   
    features={}
    for domain in feature_types:
        if domain not in valid_types:
            raise ValueError("invalid feature type: " + domain)
        else:
            if method=='peaks':
                ppi=np.diff(peaks_locs)/sampling_rate
            elif method=='troughs':
                ppi=np.diff(troughs_locs)/sampling_rate
            else:
                raise ValueError("Undefined method: " + method)

            domain_function = get_domain_function(domain)
            features.update(domain_function(sig,ppi,sampling_rate,prefix=prefix))

    return features












