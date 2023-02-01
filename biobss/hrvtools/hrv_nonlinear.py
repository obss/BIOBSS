import antropy as ant
import numpy as np
from numpy.typing import ArrayLike

# Nonlinear features
FEATURES_NL = {
    "SD1": lambda ppi: _SD1(ppi),
    "SD2": lambda ppi: _SD2(ppi),
    "SD2_SD1": lambda ppi: _SD2(ppi) / _SD1(ppi),
    "CSI": lambda ppi: (4 * _SD2(ppi)) / (4 * _SD1(ppi)),
    "CVI": lambda ppi: np.log10((4 * _SD2(ppi)) * (4 * _SD1(ppi))),
    "CSI_mofidied": lambda ppi: ((4 * _SD2(ppi)) ** 2) / (4 * _SD1(ppi)),
    "ApEn": lambda ppi: ant.app_entropy(ppi),
    "SampEn": lambda ppi: ant.sample_entropy(ppi),
}


def hrv_nl_features(ppi: ArrayLike, sampling_rate: int, prefix: str = "hrv") -> dict:
    """Calculates nonlinear hrv parameters.

    SD1: standard deviation of Poincare plot perpendicular to the line of identity
    SD2: standard deviation of Poincare plot along the line of identity
    SD2SD1:  ratio of SD2 to SD1
    CSI: cardiac stress index
    CVI: cardiac vagal index
    CSI_modified: modified cardiac stress index
    ApEn: approximate entropy of peak to peak intervals
    SampEn: sample entropy of peak to peak intervals

    Args:
        ppi (ArrayLike): Peak-to-peak interval array (miliseconds)
        prefix (str, optional): Prefix for the calculated parameters. Defaults to 'hrv'.

    Returns:
        dict: Dictionary of nonlinear hrv parameters.
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    features_nl = {}

    for key, func in FEATURES_NL.items():
        try:
            features_nl["_".join([prefix, key])] = func(ppi)
        except:
            features_nl["_".join([prefix, key])] = np.nan

    return features_nl


def _SD1(ppi):
    return np.std(((ppi[:-1] - ppi[1:]) / np.sqrt(2)), ddof=1)


def _SD2(ppi):
    return np.std(((ppi[:-1] + ppi[1:]) / np.sqrt(2)), ddof=1)
