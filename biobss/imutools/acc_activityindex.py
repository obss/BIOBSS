import warnings

import numpy as np
from numpy.typing import ArrayLike

from biobss.preprocess.signal_filter import *

DATA_TO_METRIC = {
    "PIM": ["FXYZ_modified", "UFM_modified", "UFNM", "FMpost_modified", "FMpre"],
    "ZCM": ["FXYZ", "UFM", "UFNM", "FMpost", "FMpre"],
    "TAT": ["FXYZ", "UFM", "UFNM", "FMpost", "FMpre"],
    "MAD": ["UFXYZ", "FXYZ", "UFM", "UFNM", "FMpost", "FMpre"],
    "ENMO": ["UFM"],
    "HFEN": ["SpecialXYZ", "SpecialM"],
    "AI": ["UFXYZ", "FXYZ"],
}

METRIC_FUNCTIONS = {
    "PIM": lambda sig, dim, sampling_rate, _0, _1, triaxial: _calc_pim(sig, dim, sampling_rate, triaxial),
    "ZCM": lambda sig, dim, _0, threshold, _1, triaxial: _calc_zcm(sig, dim, threshold, triaxial),
    "TAT": lambda sig, dim, sampling_rate, threshold, _0, triaxial: _calc_tat(
        sig, dim, sampling_rate, threshold, triaxial
    ),
    "MAD": lambda sig, dim, _0, _1, _2, triaxial: _calc_mad(sig, dim, triaxial),
    "ENMO": lambda sig, _0, _1, _2, _3, _4: _calc_enmo(sig),
    "HFEN": lambda sig, dim, _0, _1, _2, triaxial: _calc_hfen(sig, dim, triaxial),
    "AI": lambda sig, _0, _1, _2, baseline_variance, _4: _calc_ai(sig, baseline_variance),
}

DATASET_FUNCTIONS = {
    "UFXYZ": lambda sig_x, sig_y, sig_z, sampling_rate: generate_dataset(
        sig_x, sig_y, sig_z, sampling_rate, False, None, False, False, False
    ),
    "UFM": lambda sig_x, sig_y, sig_z, sampling_rate: generate_dataset(
        sig_x, sig_y, sig_z, sampling_rate, False, None, True, False, False
    ),
    "UFM_modified": lambda sig_x, sig_y, sig_z, sampling_rate: generate_dataset(
        sig_x, sig_y, sig_z, sampling_rate, False, None, True, False, True
    ),
    "UFNM": lambda sig_x, sig_y, sig_z, sampling_rate: generate_dataset(
        sig_x, sig_y, sig_z, sampling_rate, False, None, True, True, False
    ),
    "FXYZ": lambda sig_x, sig_y, sig_z, sampling_rate: generate_dataset(
        sig_x, sig_y, sig_z, sampling_rate, True, "pre", False, False, False
    ),
    "FXYZ_modified": lambda sig_x, sig_y, sig_z, sampling_rate: generate_dataset(
        sig_x, sig_y, sig_z, sampling_rate, True, "pre", False, False, True
    ),
    "FMpre": lambda sig_x, sig_y, sig_z, sampling_rate: generate_dataset(
        sig_x, sig_y, sig_z, sampling_rate, True, "pre", True, False, False
    ),
    "SpecialXYZ": lambda sig_x, sig_y, sig_z, sampling_rate: generate_dataset(
        sig_x, sig_y, sig_z, sampling_rate, True, "pre", False, False, False, "highpass", 4, 0.2
    ),
    "SpecialM": lambda sig_x, sig_y, sig_z, sampling_rate: generate_dataset(
        sig_x, sig_y, sig_z, sampling_rate, True, "pre", True, False, False, "highpass", 4, 0.2
    ),
    "FMpost": lambda sig_x, sig_y, sig_z, sampling_rate: generate_dataset(
        sig_x, sig_y, sig_z, sampling_rate, True, "post", True, False, False
    ),
    "FMpost_modified": lambda sig_x, sig_y, sig_z, sampling_rate: generate_dataset(
        sig_x, sig_y, sig_z, sampling_rate, True, "post", True, False, True
    ),
}


def calc_activity_index(
    accx: ArrayLike,
    accy: ArrayLike,
    accz: ArrayLike,
    signal_length: float,
    sampling_rate: float,
    metric: str,
    input_types: list = None,
    threshold: list = None,
    baseline_variance: list = None,
    triaxial: bool = False,
) -> dict:
    """Calculates the given activity index for the desired input types.

    Definitions of the activity indices are:
    Proportional Integration Method (PIM): Integration of the acceleration signal for a given epoch.
    Zero Crossing Method (ZCM) : Number of times the acceleration signal crosses a threshold.
    Time Above Threshold (TAT) : Length of timw that the acceleration signal is above a threshold.
    Mean Amplitude Deviation (MAD) : Mean absolute deviation of magnitude of acceleration values for a given epoch.
    Euclidian Norm Minus One (ENMO) : Mean  positive deviation of magnitude of acceleration values from 1g. Note that, for the magnitudes lower than 1g,
           the deviation is replaced with 0.
    High-pass Filtered Euclidian (HFEN) : Mean magnitude of highpass filtered acceleration values.
    Activity Index (AI) : Sqare root of the mean deviation of the variance of acceleration signals from the systematic noise variance for three axes.
            Note that, if the mean deviation is negative, it is replaced with 0.

        Reference:  Maczák B, Vadai G, Dér A, Szendi I, Gingl Z (2021) Detailed analysis and comparison of different activity metrics. PLOS ONE 16(12): e0261718.
                    https://doi.org/10.1371/journal.pone.0261718

    Args:
        accx (ArrayLike): Acceleration vector for the x-axis.
        accy (ArrayLike): Acceleration vector for the y-axis.
        accz (ArrayLike): Acceleration vector for the z-axis.
        signal_length (float): Signal length in seconds.
        sampling_rate (float): Sampling rate of the acceleration signal(s).
        input_types (list): Type of dataset. Depends on the preprocessing methods applied on the raw acceleration data.

            UFXYZ : Unfiltered acceleration signals ([accx, accy, accz]).
            UFM : Magnitude of unfiltered acceleration signals (magnitude([accx, accy, accz])).
            UFM_modified = Modified magnitude of unfiltered acceleration signals (absolute(UFM - integral(gravity))).
            UFNM : Normalized magnitude of unfiltered acceleration signals (normalize(magnitude([accx, accy, accz]))).
            FXYZ : Filtered acceleration signals (filter_signal([accx, accy, accz])).
            FXYZ_modified = Modified filtered acceleration signals (absolute(FXYZ)).
            FMpre : Magnitude of filtered acceleration signals  (magnitude(filter_signal([accx, accy, accz]))).
            SpecialXYZ : Filtered acceleration signals (special filter parameters).
            SpecialM : Magnitude of filtered acceleration signals (special filter parameters).
            FMpost : Filtered magnitude of acceleration signals (filter_signal(magnitude([accx, accy, accz]))).
            FMpost_modified = Modified filtered magnitude of acceleration signals (absolute(FMpost)).

        metric (str): The activity index to be calculated.
        threshold (list, optional): Threshold level in g. This parameter is required for the 'ZCM' and 'TAT' metrics. Defaults to None.
        baseline_variance (list, optional): Baseline variance, corresponding to the variance of acceleration signal at rest (no movement).
                                            This parameter is required for the 'AI' metric. Defaults to None.
        triaxial (bool, optional): Parameter to decide if triaxial metrics should be combined into a single metric or not. Defaults to False.

    Raises:
        ValueError: If the input type is not one of valid types for the desired metric.

    Returns:
        dict: A dictionary of calculated metric for the desired input types.
    """
    if not (len(accx) == len(accy) == len(accz)):
        raise ValueError("Length of input arrays (accx,accy,accz) must match!")
    if signal_length <= 0:
        raise ValueError("Signal length must be greater than 0.")
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    metric = metric.upper()

    if input_types is None:
        input_types = DATA_TO_METRIC[metric]

    valid_inputs = DATA_TO_METRIC[metric]
    metric_function = METRIC_FUNCTIONS[metric]
    act_ind = {}

    for input_type in input_types:

        if input_type not in valid_inputs:
            raise ValueError("Invalid input type for the metric to be calculated!")
        else:
            dataset_function = DATASET_FUNCTIONS[input_type]
            sig = dataset_function(accx, accy, accz, sampling_rate)
            dim = int(np.size(sig) / (signal_length * sampling_rate))

            if metric in ["ZCM", "TAT"] and threshold is None:
                warnings.warn(
                    f"Threshold level is required to calculate {metric}, but not provided. Standard deviation of the signal will be used as threshold."
                )
                threshold = _calc_threshold(sig, dim, input_type)

            act = metric_function(sig, dim, sampling_rate, threshold, baseline_variance, triaxial)
            if not triaxial:
                act_ind[input_type] = act[0]
            else:
                act_axes = {}
                act_axes["x"] = act[0]
                act_axes["y"] = act[1]
                act_axes["z"] = act[2]
                act_ind[input_type] = act_axes

    return act_ind


def generate_dataset(
    accx: ArrayLike,
    accy: ArrayLike,
    accz: ArrayLike,
    sampling_rate: float,
    filtering: bool = False,
    filtering_order: str = None,
    magnitude: bool = False,
    normalize: bool = False,
    modify: bool = False,
    filter_type: str = "bandpass",
    N: int = 2,
    f_lower: float = 0.5,
    f_upper: float = 2,
) -> ArrayLike:
    """Generates datasets by applying appropriate preprocessing steps to the raw acceleration signals.

        The datasets are:
            UFXYZ : Unfiltered acceleration signals ([accx, accy, accz]).
            UFM : Magnitude of unfiltered acceleration signals (magnitude([accx, accy, accz])).
            UFM_modified = Modified magnitude of unfiltered acceleration signals (absolute(UFM - integral(gravity))).
            UFNM : Normalized magnitude of unfiltered acceleration signals (normalize(magnitude([accx, accy, accz]))).
            FXYZ : Filtered acceleration signals (filter_signal([accx, accy, accz])).
            FXYZ_modified = Modified filtered acceleration signals (absolute(FXYZ)).
            FMpre : Magnitude of filtered acceleration signals  (magnitude(filter_signal([accx, accy, accz]))).
            SpecialXYZ : Filtered acceleration signals (special filter parameters).
            SpecialM : Magnitude of filtered acceleration signals (special filter parameters).
            FMpost : Filtered magnitude of acceleration signals (filter_signal(magnitude([accx, accy, accz]))).
            FMpost_modified = Modified filtered magnitude of acceleration signals (absolute(FMpost)).

    Args:
        accx (ArrayLike): Acceleration vector for the x-axis.
        accy (ArrayLike): Acceleration vector for the y-axis.
        accz (ArrayLike): Acceleration vector for the z-axis.
        sampling_rate (float): Sampling rate of the acceleration signal(s).
        filtering (bool, optional): Parameter to decide if filtering should be applied or not. Defaults to False.
        filtering_order (str, optional): The order of filtering, should be 'pre', 'post' or 'None'. Defaults to None.
        magnitude (bool, optional): Parameter to decide if magnitude of the signals should be calculated or not. Defaults to False.
        normalize (bool, optional): Parameter to decide if the signal should be normalized or not. Normalization refers to subtracting the gravity from the signal. Defaults to False.
        modify (bool, optional): Parameter to decide if a modification is required or not. For some of the activitiy indices,
                                 some extra modifications are required following standard preprocessing steps. These are represented as "_modified". Defaults to False.
        filter_type (str, optional): Type of the filter. Defaults to 'bandpass'.
        N (int, optional): Order of the filter. Defaults to 2.
        f_lower (float, optional): Lower cutoff frequency of the filter. Defaults to 0.5.
        f_upper (float, optional): Higher cutoff frequency of the filter. Defaults to 2.

    Raises:
        ValueError: If filtering_order is not given when filtering=True.
        ValueError: If filtering order is invalid.
        ValueError: If both normalize and filtering are selected as True.

    Returns:
        ArrayLike: The resulting preprocessed signal(s). The dimension can be either 1 or 3 depending on the type of dataset.
    """
    if not (len(accx) == len(accy) == len(accz)):
        raise ValueError("Length of input arrays (accx,accy,accz) must match!")
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    if filtering:
        if filtering_order is None:
            raise ValueError("Required parameter filtered_order.")

        elif filtering_order == "pre":
            f_x = filter_signal(
                accx, filter_type=filter_type, N=N, sampling_rate=sampling_rate, f_lower=f_lower, f_upper=f_upper
            )
            f_y = filter_signal(
                accy, filter_type=filter_type, N=N, sampling_rate=sampling_rate, f_lower=f_lower, f_upper=f_upper
            )
            f_z = filter_signal(
                accz, filter_type=filter_type, N=N, sampling_rate=sampling_rate, f_lower=f_lower, f_upper=f_upper
            )
            if magnitude:
                mag = _calc_magnitude(f_x, f_y, f_z)  # FMpre
                sig = [mag]
            else:
                if modify:
                    sig = [np.abs(f_x), np.abs(f_y), np.abs(f_z)]  # FXYZ_modified
                else:
                    sig = [f_x, f_y, f_z]  # FXYZ

        elif filtering_order == "post":
            mag = _calc_magnitude(accx, accy, accz)
            f_mag = filter_signal(
                mag, filter_type=filter_type, N=N, sampling_rate=sampling_rate, f_lower=f_lower, f_upper=f_upper
            )  # FMpost
            sig = [f_mag]
            if modify:
                sig = [np.abs(f_mag)]  # FMpost_modified

        else:
            raise ValueError(
                f"Invalid 'filtering_order' value for `generate_dataset`: {filtering_order}. Should be one of ['pre', 'post']"
            )
    else:
        if not magnitude:
            sig = [accx, accy, accz]  # UFXYZ
        else:
            mag = _calc_magnitude(accx, accy, accz)  # UFM
            if not modify and normalize:
                sig = [mag - 1]  # UFNM
            elif modify and not normalize:
                sig = [np.abs(mag - len(mag))]  # UFM_modified
            elif modify and normalize:
                raise ValueError("Both 'normalization' and 'modify' cannot be True!")
            else:
                sig = [mag]

    return sig


def _calc_threshold(sig: ArrayLike, dim: int, input_type: str) -> list:
    """Calculates the threshold level in "g", as standard deviation of the signal (It is calculated as "SD+g" for the 'UFM' dataset.)."""

    if dim == 1:
        th = np.std(sig[0])
        if input_type == "UFM":
            threshold = [th + 1]
        else:
            threshold = [th]

    elif dim == 3:
        th_x = np.std(sig[0])
        th_y = np.std(sig[1])
        th_z = np.std(sig[2])
        threshold = [th_x, th_y, th_z]

    else:
        raise ValueError("Invalid dimension!")

    return threshold


def _calc_magnitude(sig_x: ArrayLike, sig_y: ArrayLike, sig_z: ArrayLike) -> ArrayLike:
    """Calculates the magnitude signal from the axial acceleration signals."""

    return np.sqrt(np.square(sig_x) + np.square(sig_y) + np.square(sig_z))  # acc signals should be in "g"


def _calc_pim(sig: ArrayLike, dim: int, sampling_rate: float, triaxial: bool) -> list:
    """Calculates activity index using Proportional Integration Method (PIM)."""

    if dim == 1:
        pim = [np.sum(sig[0]) / sampling_rate]
    elif dim == 3:
        pim_x = np.sum(sig[0]) / sampling_rate
        pim_y = np.sum(sig[1]) / sampling_rate
        pim_z = np.sum(sig[2]) / sampling_rate

        if not triaxial:
            pim = [np.sqrt(np.square(pim_x) + np.square(pim_y) + np.square(pim_z))]
        else:
            pim = [pim_x, pim_y, pim_z]
    else:
        raise ValueError("Invalid dimension!")

    return pim


def _calc_zcm(sig: ArrayLike, dim: int, threshold: list, triaxial: bool) -> list:
    """Calculates activity index using Zero Crossing Method (ZCM)."""

    if threshold is None:
        raise ValueError("Threshold value is required for this metric.")

    else:
        if dim == 1:
            zcm = 0
            for i in range(len(sig[0]) - 1):
                if sig[0][i] < threshold[0] and sig[0][i + 1] >= threshold[0]:
                    zcm += 1
            zcm = [zcm]
        elif dim == 3:
            zcm_x = 0
            zcm_y = 0
            zcm_z = 0
            for i in range(len(sig[0]) - 1):
                if sig[0][i] < threshold[0] and sig[0][i + 1] >= threshold[0]:
                    zcm_x += 1
                if sig[1][i] < threshold[1] and sig[1][i + 1] >= threshold[1]:
                    zcm_y += 1
                if sig[2][i] < threshold[2] and sig[2][i + 1] >= threshold[2]:
                    zcm_z += 1
            if not triaxial:
                zcm = [np.sqrt(np.square(zcm_x) + np.square(zcm_y) + np.square(zcm_z))]
            else:
                zcm = [zcm_x, zcm_y, zcm_z]
        else:
            raise ValueError("Invalid dimension!")

    return zcm


def _calc_tat(sig: ArrayLike, dim: int, sampling_rate: float, threshold: list, triaxial: bool) -> list:
    """Calculates activity index using Time Above Threshold Method (TAT)."""

    if threshold is None:
        raise ValueError("Threshold value is required for this metric.")
    else:
        if dim == 1:
            tat = [len(sig[0][sig[0] >= threshold[0]]) / sampling_rate]

        elif dim == 3:
            tat_x = len(sig[0][sig[0] >= threshold[0]]) / sampling_rate
            tat_y = len(sig[1][sig[1] >= threshold[1]]) / sampling_rate
            tat_z = len(sig[2][sig[2] >= threshold[2]]) / sampling_rate
            if not triaxial:
                tat = [np.sqrt(np.square(tat_x) + np.square(tat_y) + np.square(tat_z))]
            else:
                tat = [tat_x, tat_y, tat_z]

        else:
            raise ValueError("Invalid dimension!")

    return tat


def _calc_mad(sig: ArrayLike, dim: int, triaxial: bool) -> list:
    """Calculates activity index using Mean Amplitude Deviation Method (MAD)."""

    if dim == 1:
        mad = [np.sum(np.abs(sig[0] - np.mean(sig[0]))) / len(sig[0])]
    elif dim == 3:
        mad_x = np.sum(np.abs(sig[0] - np.mean(sig[0]))) / len(sig[0])
        mad_y = np.sum(np.abs(sig[1] - np.mean(sig[1]))) / len(sig[1])
        mad_z = np.sum(np.abs(sig[2] - np.mean(sig[2]))) / len(sig[2])
        if not triaxial:
            mad = [np.sqrt(np.square(mad_x) + np.square(mad_y) + np.square(mad_z))]
        else:
            mad = [mad_x, mad_y, mad_z]
    else:
        raise ValueError("Invalid dimension!")

    return mad


def _calc_enmo(sig: ArrayLike) -> list:
    """Calculates activity index using Euclidian Norm Minus One Method (ENMO)."""

    enmo = [np.sum(sig[0][sig[0] >= 1]) / len(sig[0])]

    return enmo


def _calc_hfen(sig: ArrayLike, dim: int, triaxial: bool) -> list:
    """Calculates activity index using High-pass Filtered Euclidian Norm (HFEN)."""

    if dim == 1:
        hfen = [np.sum(sig[0]) / len(sig[0])]
    elif dim == 3:
        hfen_x = np.sum(sig[0]) / len(sig[0])
        hfen_y = np.sum(sig[1]) / len(sig[1])
        hfen_z = np.sum(sig[2]) / len(sig[2])
        if not triaxial:
            hfen = [np.sqrt(np.square(hfen_x) + np.square(hfen_y) + np.square(hfen_z))]
        else:
            hfen = [hfen_x, hfen_y, hfen_z]
    else:
        raise ValueError("Invalid dimension!")

    return hfen


def _calc_ai(sig: ArrayLike, baseline_variance: list) -> list:
    """Calculates activity index using Activity Index Method (AI)."""

    if baseline_variance is None:
        raise ValueError("Baseline variance is required for this metric.")
    else:
        x_ = np.var(sig[0]) - baseline_variance[0]
        y_ = np.var(sig[1]) - baseline_variance[1]
        z_ = np.var(sig[2]) - baseline_variance[2]
        ai = [np.sqrt(max([np.mean([x_, y_, z_]), 0]))]

    return ai
