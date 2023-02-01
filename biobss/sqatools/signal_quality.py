import math
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

# Constants to check for physiological viability and morphological features.
HR_MIN = 40
HR_MAX = 180
PP_MAX = 3
MAX_PP_RATIO = 2.2
MIN_SPD = 0.08
MAX_SPD = 0.49
SP_DP_RATIO = 1.1
MIN_PWD = 0.27
MAX_PWD = 2.4
MAX_VAR_DUR = 300
MAX_VAR_AMP = 400
CORR_TH = 0.9


def detect_clipped_segments(sig: ArrayLike, threshold_pos: float, threshold_neg: float = None) -> list:
    """Detects clipped segments in a signal.

    Args:
        sig (ArrayLike): Signal to be analyzed (ECG or PPG).
        threshold_pos (float): Threshold for positive clipping
        threshold_neg (float, optional): Threshold for negative clipping. Defaults to None.

    Returns:
        list: Dictionary of boundaries of clipped segments.
    """

    if threshold_neg is None:
        threshold_neg = -threshold_pos

    start_indices = []
    end_indices = []
    in_clipped_segment = False

    for i, value in enumerate(sig):

        if value >= threshold_pos or value <= threshold_neg:
            if not in_clipped_segment:
                # Start of a new clipped segment
                start_indices.append(i)
                in_clipped_segment = True
        else:
            if in_clipped_segment:
                # End of a clipped segment
                end_indices.append(i - 1)
                in_clipped_segment = False

    if in_clipped_segment:
        # The last segment extends until the end of the signal
        end_indices.append(len(sig) - 1)

    return list(zip(start_indices, end_indices))


def detect_flatline_segments(sig: ArrayLike, min_duration: float, change_threshold: float) -> list:
    """Detects flatline segments in a signal.

    Args:
        sig (ArrayLike): Signal to be analyzed (ECG or PPG).
        min_duration (float): Mimimum duration of flat segments for flatline detection.
        change_threshold (float): Threshold for change in signal amplitude.

    Returns:
        list: List of boundaries of flatline segments.
    """

    start_indices = []
    end_indices = []
    in_flatline_segment = False

    for i in range(1, len(sig)):
        change = abs(sig[i] - sig[i - 1])

        if change <= change_threshold and sig[i] != max(sig) and sig[i] != min(sig):
            if not in_flatline_segment:
                # Start of a new flatline segment
                start_indices.append(i - 1)
                in_flatline_segment = True
        else:
            if in_flatline_segment:
                # End of a flatline segment
                end_indices.append(i - 1)
                in_flatline_segment = False

    if in_flatline_segment:
        # The last segment extends until the end of the signal
        end_indices.append(len(sig) - 1)

    # Filter segments by duration
    durations = [end - start + 1 for start, end in zip(start_indices, end_indices)]
    start_indices = [start for start, duration in zip(start_indices, durations) if duration >= min_duration]
    end_indices = [end for end, duration in zip(end_indices, durations) if duration >= min_duration]

    return list(zip(start_indices, end_indices))


def check_phys(peaks_locs: ArrayLike, sampling_rate: float) -> dict:
    """Checks for physiological viability.

    Rule 1: Average HR should be between 40-180 bpm (up to 300 bpm in the case of exercise)
    Rule 2: Maximum P-P interval: 1.5 seconds. Allowing for a single missing beat, it is 3 seconds
    Rule 3: Maximum P-P interval / minimum P-P interval ratio: 10 of the signal length for a short signal.
            For 10 seconds signal, it is 1.1; allowing for a single missing beat, it is 2.2

    Args:
        peaks_locs (ArrayLike): Array of peak locations.
        sampling_rate (float): Sampling rate of the input signal.

    Returns:
        dict: Dictionary of decisions.
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    info = {}

    # Rule 1: Average HR should be between 40-180 bpm (up to 300 bpm in the case of exercise)
    intervals = np.diff(peaks_locs) / sampling_rate
    HR_mean = 60 / np.mean(intervals)

    if HR_mean < HR_MIN or HR_mean > HR_MAX:
        info["Rule 1"] = False
    else:
        info["Rule 1"] = True

    # Rule 2: Maximum P-P interval: 1.5 seconds. Allowing for a single missing beat, it is 3 seconds
    if np.size(np.where(intervals > PP_MAX)) > 0:
        info["Rule 2"] = False
    else:
        info["Rule 2"] = True

    # Rule 3: Maximum P-P interval / minimum P-P interval ratio: 10 of the signal length for a short signal.
    # For 10 seconds signal, it is 1.1; allowing for a single missing beat, it is 2.2
    if (intervals.max() / intervals.min()) > MAX_PP_RATIO:
        info["Rule 3"] = False
    else:
        info["Rule 3"] = True

    return info


def check_morph(sig: ArrayLike, peaks_locs: ArrayLike, troughs_locs: ArrayLike, sampling_rate: float) -> dict:
    """Checks for ranges of morphological features.

    Rule 1: Systolic phase duration(rise time): 0.08 to 0.49 s
    Rule 2: Ratio of systolic phase duration to diastolic phase duration: max 1.1
    Rule 3: Pulse wave duration: 0.27 to 2.4 s
    Rule 4: Variation in PWD and SP: 33-300%
    Rule 5: Variation in PWA: 25-400% (Pulse wave amplitude: a threshold which was set heuristically)

    Args:
        peaks_locs (Array): Array of peak locations.
        peaks_amps (Array): Array of peak amplitudes.
        troughs_locs (Array): Array of trough locations.
        troughs_amps (Array): Array of trough amplitudes.
        sampling_rate (float): Sampling rate of the input signal.

    Returns:
        dict: Dictionary of decisions.
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    if len(peaks_locs) != len(troughs_locs) - 1:
        raise ValueError("Number of peaks and troughs are not compatible!")

    peaks_amps = sig[peaks_locs]
    troughs_amps = sig[troughs_locs]

    info = {}

    # Rule 1
    SP = (peaks_locs - troughs_locs[:-1]) / sampling_rate

    if np.size(np.where(SP < MIN_SPD)) > 0 or np.size(np.where(SP > MAX_SPD)) > 0:
        info["Rule 1"] = False
    else:
        info["Rule 1"] = True

    # Rule 2:
    DP = (troughs_locs[1:] - peaks_locs) / sampling_rate  # Diastolic phase duration
    SP_DP = SP / DP

    if np.size(np.where(SP_DP > SP_DP_RATIO)) > 0:
        info["Rule 2"] = False
    else:
        info["Rule 2"] = True

    # Rule 3:
    PWD = np.diff(troughs_locs) / sampling_rate

    if np.size(np.where(PWD < MIN_PWD)) > 0 or np.size(np.where(PWD > MAX_PWD)) > 0:
        info["Rule 3"] = False
    else:
        info["Rule 3"] = True

    # Rule 4:
    var_SP = (np.max(peaks_amps) - np.min(peaks_amps)) / np.min(peaks_amps) * 100
    var_PWD = (np.max(PWD) - np.min(PWD)) / np.min(PWD) * 100

    if (var_SP > MAX_VAR_DUR) or (var_PWD > MAX_VAR_DUR):
        info["Rule 4"] = False
    else:
        info["Rule 4"] = True

    # Rule 5:
    PWA = peaks_amps - troughs_amps[:-1]
    var_PWA = (np.max(PWA) - np.min(PWA)) / np.min(PWA) * 100

    if var_PWA > MAX_VAR_AMP:
        info["Rule 5"] = False
    else:
        info["Rule 5"] = True

    return info


def template_matching(sig: ArrayLike, peaks_locs: ArrayLike, corr_th: float = CORR_TH) -> Tuple[float, bool]:
    """Applies template matching method for signal quality assessment.

    Args:
        sig (ArrayLike): Signal to be analyzed.
        peaks_locs (ArrayLike): Peak locations (Systolic peaks for PPG signal, R peaks for ECG signal).
        corr_th (float, optional): Threshold for the correlation coefficient above which the signal is considered to be valid. Defaults to CORR_TH.

    Returns:
        Tuple[float,bool]: Correlation coefficient and the decision
    """
    if corr_th <= 0:
        raise ValueError("Threshold for the correlation coefficient must be greater than 0.")

    wl = np.median(np.diff(peaks_locs))
    waves = np.empty((0, 2 * math.floor(wl / 2) + 1))
    nofwaves = np.size(peaks_locs)

    for i in range((nofwaves)):
        wave_st = peaks_locs[i] - math.floor(wl / 2)
        wave_end = peaks_locs[i] + math.floor(wl / 2)
        wave = []

        if wave_st < 0:
            wave = sig[:wave_end]
            for _ in range(-wave_st + 1):
                wave = np.insert(wave, 0, wave[0])

        elif wave_end > len(sig) - 1:
            wave = sig[wave_st - 1 :]
            for _ in range(wave_end - len(sig)):
                wave = np.append(wave, wave[-1])

        else:
            wave = sig[wave_st : wave_end + 1]

        waves = np.vstack([waves, wave])

    sig_temp = np.mean(waves, axis=0)

    ps = np.array([])
    for j in range(np.size(peaks_locs)):
        p = np.corrcoef(waves[j], sig_temp, rowvar=True)
        ps = np.append(ps, p[0][1])

    if np.size(np.where(ps < corr_th)) > 0:
        result = False

    else:
        result = True

    return ps, result
