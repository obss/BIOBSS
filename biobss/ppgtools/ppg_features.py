from typing import Callable

from numpy.typing import ArrayLike

from biobss.ppgtools.ppg_freqdomain import *
from biobss.ppgtools.ppg_statistical import *
from biobss.ppgtools.ppg_timedomain import *


def get_domain_function(domain: str) -> Callable:

    if domain == "Time":
        return ppg_time_features
    elif domain == "Freq":
        return ppg_freq_features
    elif domain == "Stat":
        return ppg_stat_features
    else:
        raise ValueError("Unknown domain:", domain)


def get_ppg_features(
    sig: ArrayLike,
    sampling_rate: float,
    fiducials: dict = None,
    input_types: list = ["cycle", "segment"],
    feature_domain: dict = {"cycle": ["Time", "Stat"], "segment": ["Stat", "Freq", "Time"]},
    prefix: str = "ppg",
    **kwargs
) -> dict:
    """Calculates PPG features.

    Args:
        sig (ArrayLike): PPG signal.
        sampling_rate (float): Sampling rate of the PPG signal (Hz).
        fiducials (dict, optional): PPG fiducials. Defaults to None.
        input_types (list, optional): Input types. It can be a list of 'cycle' and 'segment'. Defaults to ['cycle', 'segment'].
        feature_domain (_type_, optional): Domain to calculate features. It should be provided for each input type seperately. Defaults to {'cycle':['Time','Stat'], 'segment':['Stat','Freq','Time']}.
        prefix (str, optional): Prefix for the features. Defaults to 'ppg'.

    Raises:
        ValueError: If sampling rate is not greater than 0.
        ValueError: If keyword arguments are missing.

    Returns:
        dict: Dictionary of PPG features
    """

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    input_types = [x.lower() for x in input_types]

    features = {}
    if "cycle" in input_types:
        feature_domain["cycle"] = [x.capitalize() for x in feature_domain["cycle"]]

        if all(k in kwargs.keys() for k in ("peaks_locs", "troughs_locs")):
            features_cycle = from_cycles(
                sig,
                sampling_rate=sampling_rate,
                fiducials=fiducials,
                peaks_locs=kwargs["peaks_locs"],
                troughs_locs=kwargs["troughs_locs"],
                feature_types=feature_domain["cycle"],
                prefix=prefix,
            )
            features.update(features_cycle)
        else:
            raise ValueError("Missing keyword arguments for the input_type: 'cycle'!")

    if "segment" in input_types:
        feature_domain["segment"] = [x.capitalize() for x in feature_domain["segment"]]

        features_segment = from_segment(
            sig, sampling_rate=sampling_rate, feature_types=feature_domain["segment"], prefix=prefix
        )
        features.update(features_segment)

    return features


def from_cycles(
    sig: ArrayLike,
    peaks_locs: ArrayLike,
    troughs_locs: ArrayLike,
    sampling_rate: float,
    fiducials: dict = None,
    feature_types: ArrayLike = ["Time", "Stat"],
    prefix: str = "ppg",
) -> dict:
    """Calculates cycle-based PPG features.

    Args:
        sig (ArrayLike): PPG signal segment to be analyzed
        peaks_locs (ArrayLike): PPG peak locations
        troughs_locs (ArrayLike): PPG trough locations
        sampling_rate (float): Sampling rate of the PPG signal.
        fiducials (dict, optional): PPG fiducials. Defaults to None.
        feature_types (ArrayLike, optional): Types of features to be calculated. Defaults to ['Time','Stat'].
        prefix (str, optional): Prefix for signal type. Defaults to 'signal'.

    Raises:
        ValueError: If elements of feature_types are not 'Time' or 'Stat'.

    Returns:
        dict: Dictionary of calculated features.
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    if len(peaks_locs) != len(troughs_locs) - 1:
        raise ValueError("Lengths of peak and trough arrays do not match!")

    feature_types = [x.capitalize() for x in feature_types]

    valid_types = ["Time", "Stat"]

    peaks_amp = sig[peaks_locs]
    troughs_amp = sig[troughs_locs]

    features = {}
    for domain in feature_types:
        if domain not in valid_types:
            raise ValueError("Invalid feature type: " + domain)
        else:
            domain_function = get_domain_function(domain)
            features.update(
                domain_function(
                    sig=sig,
                    sampling_rate=sampling_rate,
                    fiducials=fiducials,
                    input_types=["cycle"],
                    prefix=prefix,
                    peaks_locs=peaks_locs,
                    peaks_amp=peaks_amp,
                    troughs_locs=troughs_locs,
                    troughs_amp=troughs_amp,
                )
            )

    return features


def from_segment(
    sig: ArrayLike, sampling_rate: float, feature_types: ArrayLike = ["Stat", "Freq", "Time"], prefix: str = "signal"
) -> dict:
    """Calculates segment-based PPG features.

    Args:
        sig (ArrayLike): PPG signal segment to be analyzed
        sampling_rate (float): Sampling rate of the PPG signal.
        feature_types (ArrayLike, optional): Types of features to be calculated. Defaults to ['Stat','Freq','Time'].
        prefix (str, optional): Prefix for signal type. Defaults to 'signal'.

    Raises:
        ValueError: if elements of feature_types are not 'Time', 'Stat' or 'Freq'.

    Returns:
        dict: Dictionary of calculated features.
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    feature_types = [x.capitalize() for x in feature_types]

    valid_types = ["Time", "Stat", "Freq"]
    features = {}
    for domain in feature_types:
        if domain not in valid_types:
            raise ValueError("invalid feature type: " + domain)
        else:
            domain_function = get_domain_function(domain)
            features.update(
                domain_function(sig=sig, sampling_rate=sampling_rate, input_types=["segment"], prefix=prefix)
            )

    return features
