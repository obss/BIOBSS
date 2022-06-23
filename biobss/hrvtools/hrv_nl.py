import numpy as np
from numpy.typing import ArrayLike
import antropy as ant

#Nonlinear features
FEATURES_NL = {
'SD1': lambda ppi: _SD1(ppi),
'SD2' : lambda ppi: _SD2(ppi),
'SD1SD2' : lambda ppi: _SD1(ppi)/_SD2(ppi),
'CSI' : lambda ppi: (4* _SD2(ppi)) / (4* _SD1(ppi)),
'CVI' : lambda ppi: np.log10((4* _SD2(ppi))*(4*_SD1(ppi))),
'CSI_mofidied' : lambda ppi: (4*_SD2(ppi)**2)/(4*_SD1(ppi)),
'ApEn' : lambda ppi: ant.app_entropy(ppi.tolist()),
'SampEn': lambda ppi: ant.sample_entropy(ppi.tolist()),
}

def hrv_nl_features(ppi: ArrayLike, prefix: str='hrv') -> dict:
    """Calculates nonlinear hrv parameters.

    SD1: the standard deviation of the Poincare plot perpendicular to the line of identity
    SD2: the standard deviation of the Poincare plot along to the line of identity
    SD2/SD1: the ratio of SD2 to SD1
    ApEn: approximate entropy of intervals
    SampEn: Sample entropy of intervals
 
    Args:
        ppi (ArrayLike): Peak-to-peak interval array
        prefix (str, optional): Prefix for the calculated parameters. Defaults to 'hrv'.

    Returns:
        dict: Dictionary of nonlinear hrv parameters.
    """

    features_nl={}

    for key,func in FEATURES_NL.items():
        features_nl["_".join([prefix, key])]=func(ppi)

    return features_nl


def _SD1(ppi):
    return np.sqrt( 0.5 * (np.std(np.diff(ppi), ddof=1) ** 2))*1000

def _SD2(ppi):
    return np.sqrt(2 * (np.std(ppi, ddof=1) ** 2) - 0.5 * (np.std(np.diff(ppi), ddof=1) ** 2))*1000



