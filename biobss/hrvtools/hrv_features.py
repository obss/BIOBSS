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


def get_hrv_features(sampling_rate: float, input_type='ppi', peaks_locs=None, troughs_locs=None,  ppi=None, feature_types: ArrayLike=['Freq','Time','Nonlinear'], prefix: str='hrv') -> dict:
    """Calculates hrv parameters

    Args:
        sampling_rate (float): Sampling rate of the ppg signal.
        input_type (str, optional): Input type for the analyses. Should be 'ppi', 'peaks' or 'troughs'. Defaults to 'ppi'.
                                    Depending on the input type, corresponding input array should be provided. 
        peaks_locs (_type_, optional): Peak locations of the ppg signal. Defaults to None.
        troughs_locs (_type_, optional): Onset locations of the ppg signal. Defaults to None.
        ppi (_type_, optional): Peak-to-peak intervals of the ppg signal. Defaults to None.
        feature_types (ArrayLike, optional): List of the type of hrv parameters to be calculated. Defaults to ['Freq','Time','Nonlinear'].
        prefix (str, optional): Prefix for the calculated parameters. Defaults to 'hrv'.

    Raises:
        ValueError: If elements of feature_types are not 'Freq', 'Time' or 'Nonlinear'.
        ValueError: If 'ppi' is None although the input type is 'ppi'.
        ValueError: If 'peaks_locs' is None although the input type is 'peaks'.
        ValueError: If 'trough_locs' is None although the input type is 'troughs'.
        ValueError: If the input_type is not 'ppi', 'peaks' or 'troughs.

    Returns:
        dict: Dictionary of calculated hrv parameters.
    """

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











