from numpy.typing import ArrayLike
from typing import Callable
import numpy as np

from biobss.hrvtools.hrv_fd import hrv_freq_features
from biobss.hrvtools.hrv_td import hrv_time_features
from biobss.hrvtools.hrv_nl import hrv_nl_features

def get_domain_function(domain:str) -> Callable:

    if domain == "Time":
        return hrv_time_features
    elif domain == "Freq":
        return hrv_freq_features
    elif domain == "Nonlinear":
        return hrv_nl_features
    else:
        raise ValueError("Unknown domain:", domain)  


def get_hrv_features(sig: ArrayLike, sampling_rate: float, input_type='ppi', peaks_locs=None, troughs_locs=None,  ppi=None, feature_types: ArrayLike=['Freq','Time','Nonlinear'], prefix: str='hrv') -> dict:

    valid_types=['Time','Freq','Nonlinear']   
    features={}
    for domain in feature_types:
        if domain not in valid_types:
            raise ValueError("invalid feature type: " + domain)
        else:
            if input_type=='ppi':
                if ppi is None:
                    raise ValueError("The argument 'ppi' is required.")

            elif input_type=='peaks':
                if peaks_locs is None:
                    raise ValueError("The argument 'peaks_locs' is required.")
                else:                
                    ppi=np.diff(peaks_locs)/sampling_rate

            elif input_type=='troughs':
                if troughs_locs is None:
                    raise ValueError("The argument 'troughs_locs' is required.")
                else:
                    ppi=np.diff(troughs_locs)/sampling_rate
                    
            else:
                raise ValueError("Undefined input type: " + input_type)

            domain_function = get_domain_function(domain)
            features.update(domain_function(ppi,prefix=prefix))

    return features












