import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from biobss.ppgtools.ppg_peaks import *

# Time domain features
FEATURES_TIME_CYCLE = {
    "a_S": lambda sig, _0, locs_S, _1, _2, _3: np.mean(sig[locs_S]),
    "t_S": lambda _0, sampling_rate, locs_S, locs_O, _1, _2: np.mean((locs_S - locs_O[:-1]) / sampling_rate),
    "t_C": lambda _0, sampling_rate, _1, locs_O, _2, _3: np.mean(np.diff(locs_O) / sampling_rate),
    "DW": lambda _0, sampling_rate, locs_S, locs_O, _1, _2: np.mean((locs_O[1:] - locs_S) / sampling_rate),
    "SW_10": lambda sig, sampling_rate, locs_S, locs_O, _0, _1: np.mean(
        _calculate_SW(sig, locs_S, locs_O, sampling_rate, 0.1)
    ),
    "SW_25": lambda sig, sampling_rate, locs_S, locs_O, _0, _1: np.mean(
        _calculate_SW(sig, locs_S, locs_O, sampling_rate, 0.25)
    ),
    "SW_33": lambda sig, sampling_rate, locs_S, locs_O, _0, _1: np.mean(
        _calculate_SW(sig, locs_S, locs_O, sampling_rate, 0.33)
    ),
    "SW_50": lambda sig, sampling_rate, locs_S, locs_O, _0, _1: np.mean(
        _calculate_SW(sig, locs_S, locs_O, sampling_rate, 0.5)
    ),
    "SW_66": lambda sig, sampling_rate, locs_S, locs_O, _0, _1: np.mean(
        _calculate_SW(sig, locs_S, locs_O, sampling_rate, 0.66)
    ),
    "SW_75": lambda sig, sampling_rate, locs_S, locs_O, _0, _1: np.mean(
        _calculate_SW(sig, locs_S, locs_O, sampling_rate, 0.75)
    ),
    "DW_10": lambda sig, sampling_rate, locs_S, locs_O, _0, _1: np.mean(
        _calculate_DW(sig, locs_S, locs_O, sampling_rate, 0.1)
    ),
    "DW_25": lambda sig, sampling_rate, locs_S, locs_O, _0, _1: np.mean(
        _calculate_DW(sig, locs_S, locs_O, sampling_rate, 0.25)
    ),
    "DW_33": lambda sig, sampling_rate, locs_S, locs_O, _0, _1: np.mean(
        _calculate_DW(sig, locs_S, locs_O, sampling_rate, 0.33)
    ),
    "DW_50": lambda sig, sampling_rate, locs_S, locs_O, _0, _1: np.mean(
        _calculate_DW(sig, locs_S, locs_O, sampling_rate, 0.5)
    ),
    "DW_66": lambda sig, sampling_rate, locs_S, locs_O, _0, _1: np.mean(
        _calculate_DW(sig, locs_S, locs_O, sampling_rate, 0.66)
    ),
    "DW_75": lambda sig, sampling_rate, locs_S, locs_O, _0, _1: np.mean(
        _calculate_DW(sig, locs_S, locs_O, sampling_rate, 0.75)
    ),
    "DW_SW_10": lambda sig, sampling_rate, locs_S, locs_O, _0, _1: np.mean(
        _calculate_DW_SW(sig, locs_S, locs_O, sampling_rate, 0.1)
    ),
    "DW_SW_25": lambda sig, sampling_rate, locs_S, locs_O, _0, _1: np.mean(
        _calculate_DW_SW(sig, locs_S, locs_O, sampling_rate, 0.25)
    ),
    "DW_SW_33": lambda sig, sampling_rate, locs_S, locs_O, _0, _1: np.mean(
        _calculate_DW_SW(sig, locs_S, locs_O, sampling_rate, 0.33)
    ),
    "DW_SW_50": lambda sig, sampling_rate, locs_S, locs_O, _0, _1: np.mean(
        _calculate_DW_SW(sig, locs_S, locs_O, sampling_rate, 0.50)
    ),
    "DW_SW_66": lambda sig, sampling_rate, locs_S, locs_O, _0, _1: np.mean(
        _calculate_DW_SW(sig, locs_S, locs_O, sampling_rate, 0.66)
    ),
    "DW_SW_75": lambda sig, sampling_rate, locs_S, locs_O, _0, _1: np.mean(
        _calculate_DW_SW(sig, locs_S, locs_O, sampling_rate, 0.75)
    ),
    "PR_mean": lambda _0, sampling_rate, locs_S, _1, _2, _3: 60 / np.mean(np.diff(locs_S) / sampling_rate),
    "a_D": lambda sig, _0, _1, _2, locs_D, _3: np.mean(sig[locs_D]),
    "t_D": lambda _0, sampling_rate, _1, locs_O, locs_D, _2: np.mean((locs_D - locs_O[:-1]) / sampling_rate),
    "r_D": lambda sig, sampling_rate, _0, locs_O, locs_D, _1: np.mean(
        sig[locs_D] / ((locs_D - locs_O[:-1]) / sampling_rate)
    ),
    "a_N": lambda sig, _0, _1, _2, _3, locs_N: np.mean(sig[locs_N]),
    "t_N": lambda _0, sampling_rate, _1, locs_O, _2, locs_N: np.mean((locs_N - locs_O[:-1]) / sampling_rate),
    "r_N": lambda sig, sampling_rate, _0, locs_O, _1, locs_N: np.mean(
        sig[locs_N] / ((locs_N - locs_O[:-1]) / sampling_rate)
    ),
    "dT": lambda _0, sampling_rate, locs_S, _1, locs_D, _2: np.mean((locs_D - locs_S) / sampling_rate),
    "r_D_NC": lambda sig, sampling_rate, _0, locs_O, locs_D, locs_N: np.mean(
        sig[locs_D] / ((np.diff(locs_O) / sampling_rate) - ((locs_N - locs_O[:-1]) / sampling_rate))
    ),
    "r_N_NC": lambda sig, sampling_rate, _0, locs_O, _1, locs_N: np.mean(
        sig[locs_N] / ((np.diff(locs_O) / sampling_rate) - ((locs_N - locs_O[:-1]) / sampling_rate))
    ),
    "a_N_S": lambda sig, _0, locs_S, _1, _2, locs_N: np.mean(sig[locs_N] / sig[locs_S]),
    "AI": lambda sig, _0, locs_S, _1, locs_D, _2: np.mean(sig[locs_D] / sig[locs_S]),
    "AI_2": lambda sig, _0, locs_S, _1, locs_D, _2: np.mean((sig[locs_S] - sig[locs_D]) / sig[locs_S]),
}

FEATURES_TIME_SEGMENT = {
    "zcr": lambda sig, _0: _calculate_zcr(sig),
    "snr": lambda sig, _0: _calculate_snr(sig),
}


def ppg_time_features(
    sig: ArrayLike, sampling_rate: float, input_types: str, fiducials: dict = None, prefix: str = "ppg", **kwargs
) -> dict:
    """Calculates time-domain features.

    Cycle-based features:
        a_S: Mean amplitude of the systolic peaks
        t_S: Mean systolic peak duration
        t_C: Mean cycle duration
        DW: Mean diastolic peak duration
        SW_10: The systolic peak duration at 10% of systolic amplitude
        SW_25: The systolic peak duration at 25% of systolic amplitude
        SW_33: The systolic peak duration at 33% of systolic amplitude
        SW_50: The systolic peak duration at 50% of systolic amplitude
        SW_66: The systolic peak duration at 66% of systolic amplitude
        SW_75: The systolic peak duration at 75% of systolic amplitude
        DW_10: The diastolic peak duration at 10% of systolic amplitude
        DW_25: The diastolic peak duration at 25% of systolic amplitude
        DW_33: The diastolic peak duration at 33% of systolic amplitude
        DW_50: The diastolic peak duration at 50% of systolic amplitude
        DW_66: The diastolic peak duration at 66% of systolic amplitude
        DW_75: The diastolic peak duration at 75% of systolic amplitude
        DW_SW_10: The ratio of diastolic peak duration to systolic peak duration at 10% of systolic amplitude
        DW_SW_25: The ratio of diastolic peak duration to systolic peak duration at 25% of systolic amplitude
        DW_SW_33: The ratio of diastolic peak duration to systolic peak duration at 33% of systolic amplitude
        DW_SW_50: The ratio of diastolic peak duration to systolic peak duration at 50% of systolic amplitude
        DW_SW_66: The ratio of diastolic peak duration to systolic peak duration at 66% of systolic amplitude
        DW_SW_75: The ratio of diastolic peak duration to systolic peak duration at 75% of systolic amplitude
        PR_mean: Mean pulse rate
        a_D: Mean amplitude of the diastolic peaks
        t_D: Mean difference between diastolic peak and onset
        r_D: Mean ratio of the diastolic peak amplitude to diastolic peak duration
        a_N: Mean amplitude of the dicrotic notchs
        t_N: Mean dicrotic notch duration
        r_N: Mean ratio of the dicrotic notch amplitude to dicrotic notch duration
        dT: Mean duration from systolic to diastolic peaks
        r_D_NC: Mean ratio of diastolic peak amplitudes to difference between ppg wave duration and dictoric notch duration
        r_N_NC: Mean ratio of dicrotic notch amplitudes to difference between ppg wave duration and dictoric notch duration
        a_N_S: Mean ratio of dicrotic notch amplitudes to systolic peak amplitudes
        AI: Mean ratio of diastolic peak amplitudes to systolic peak amplitudes
        AI_2: Mean ratio of difference between systolic and diastolic peak amplitudes to systolic peak amplitudes

    Segment-based features:
        zcr: Zero crossing rate
        snr: Signal to noise ratio

    Args:
        sig (ArrayLike): Signal to be analyzed.
        sampling_rate (float): Sampling rate of the signal (Hz).
        input_types (str): Type of feature calculation, should be 'segment' or 'cycle'.
        fiducials (dict, optional): Dictionary of fiducial point locations. Defaults to None.
        prefix (str, optional): Prefix for signal type. Defaults to 'ppg'.

    Kwargs:
        peaks_locs (ArrayLike): Array of peak locations
        troughs_locs (ArrayLike): Array of trough locations

    Raises:
        ValueError: If sampling rate is not greater than 0.
        ValueError: If PPG onset locations is not provided.
        ValueError: If PPG S-peak locations is not provided.
        ValueError: If Type is not 'cycle' or 'segment'.

    Returns:
        dict: Dictionary of calculated features.
    """

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    input_types = [x.lower() for x in input_types]

    features_time = {}
    for type in input_types:
        if type == "cycle":
            feature_list = FEATURES_TIME_CYCLE.copy()

            if fiducials is not None:
                fiducial_names = ["O_waves", "S_waves", "D_waves", "N_waves"]
                fiducials = {key: fiducials.get(key, []) for key in fiducial_names}

                locs_O = fiducials["O_waves"]
                locs_S = fiducials["S_waves"]
                locs_D = fiducials["D_waves"]
                locs_N = fiducials["N_waves"]

                locs_S, _ = correct_missing_duplicate_peaks(locs_valleys=locs_O, locs_peaks=locs_S, peaks=sig[locs_S])
                locs_D, _ = correct_missing_duplicate_peaks(locs_valleys=locs_O, locs_peaks=locs_D, peaks=sig[locs_D])
                locs_N, _ = correct_missing_duplicate_peaks(locs_valleys=locs_O, locs_peaks=locs_N, peaks=sig[locs_N])

            else:
                locs_O = kwargs["troughs_locs"]
                locs_S = kwargs["peaks_locs"]
                locs_D = np.array([])
                locs_N = np.array([])

                locs_S, _ = correct_missing_duplicate_peaks(locs_valleys=locs_O, locs_peaks=locs_S, peaks=sig[locs_S])

            if len(locs_O) == 0:
                raise ValueError("PPG onset locations must be provided to calculate cycle-based features.")

            if len(locs_S) == 0:
                raise ValueError("PPG S peak locations must be provided to calculate cycle-based features.")

            if len(locs_D) == 0:
                D_features = ["a_D", "t_D", "r_D", "dT", "r_D_NC", "r_N_NC", "AI", "AI_2"]
                [feature_list.pop(key, None) for key in D_features]

            if len(locs_N) == 0:
                N_features = ["a_N", "t_N", "r_N", "r_D_NC", "r_N_NC", "a_N_S"]
                [feature_list.pop(key, None) for key in N_features]

            for key, func in feature_list.items():
                try:
                    features_time["_".join([prefix, key])] = func(sig, sampling_rate, locs_S, locs_O, locs_D, locs_N)
                except:
                    features_time["_".join([prefix, key])] = np.nan

        elif type == "segment":
            for key, func in FEATURES_TIME_SEGMENT.items():
                try:
                    features_time["_".join([prefix, key])] = func(sig, sampling_rate)
                except:
                    features_time["_".join([prefix, key])] = np.nan

        else:
            raise ValueError("Type should be 'cycle' or 'segment'.")

    return features_time


def _calculate_SW(
    sig: ArrayLike, peaks_locs: ArrayLike, troughs_locs: ArrayLike, sampling_rate: float, ratio: float
) -> float:
    """Calculates systolic phase duration of the PPG waveform."""
    peaks_amp = sig[peaks_locs]
    troughs_amp = sig[troughs_locs]

    SWs = []
    for c in range(len(troughs_locs) - 1):
        sys_amp = peaks_amp[c] - troughs_amp[c]
        thresh = (ratio * sys_amp) + troughs_amp[c]
        ind = np.where(sig[troughs_locs[c] : peaks_locs[c]] >= thresh)
        SWs.append(len(ind[0]) / sampling_rate)

    return np.array(SWs)


def _calculate_DW(
    sig: ArrayLike, peaks_locs: ArrayLike, troughs_locs: ArrayLike, sampling_rate: float, ratio: float
) -> float:
    """Calculates diastolic phase duration of the waveform."""
    peaks_amp = sig[peaks_locs]
    troughs_amp = sig[troughs_locs]

    DWs = []
    for c in range(len(troughs_locs) - 1):
        sys_amp = peaks_amp[c] - troughs_amp[c]
        thresh = (ratio * sys_amp) + troughs_amp[c]
        ind = np.where(sig[peaks_locs[c] : troughs_locs[c + 1]] >= thresh)
        DWs.append(len(ind[0]) / sampling_rate)

    return np.array(DWs)


def _calculate_DW_SW(
    sig: ArrayLike, peaks_locs: ArrayLike, troughs_locs: ArrayLike, sampling_rate: float, ratio: float
) -> float:
    """Calculates the ratio of diastolic phase duration of the waveform to systolic phase duration of the waveform."""

    dw = _calculate_DW(sig, peaks_locs=peaks_locs, troughs_locs=troughs_locs, sampling_rate=sampling_rate, ratio=ratio)
    sw = _calculate_SW(sig, peaks_locs=peaks_locs, troughs_locs=troughs_locs, sampling_rate=sampling_rate, ratio=ratio)

    return dw / sw


def _calculate_zcr(sig: ArrayLike) -> float:
    """Calculates zero crossing rate, defined as number of zero-crossings to signal length."""

    sig_ = sig - np.mean(sig)
    numZeroCrossing = len(np.where(np.diff(np.sign(sig_)))[0])

    return numZeroCrossing / len(sig_)


def _calculate_snr(sig: ArrayLike) -> float:
    """Calculates signal to noise ratio."""

    mn_sig = np.mean(sig)
    std_sig = np.std(sig)
    snratio = np.where(std_sig == 0, 0, mn_sig / std_sig).item()

    return snratio
