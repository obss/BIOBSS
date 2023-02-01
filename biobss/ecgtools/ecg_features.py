import numpy as np
from numpy.typing import ArrayLike

# Morphological features from R-peak locations
FEATURES_RPEAKS = {
    "a_R": lambda sig, _0, peaks_locs, beatno: sig[peaks_locs[beatno]],
    "RR0": lambda _0, sampling_rate, peaks_locs, beatno: _get_RR_interval(peaks_locs, sampling_rate, beatno, -1),
    "RR1": lambda _0, sampling_rate, peaks_locs, beatno: _get_RR_interval(peaks_locs, sampling_rate, beatno, 0),
    "RR2": lambda _0, sampling_rate, peaks_locs, beatno: _get_RR_interval(peaks_locs, sampling_rate, beatno, 1),
    "RRm": lambda _0, sampling_rate, peaks_locs, beatno: _get_mean_RR(peaks_locs, sampling_rate, beatno),
    "RR_0_1": lambda _0, sampling_rate, peaks_locs, beatno: _get_RR_interval(peaks_locs, sampling_rate, beatno, -1)
    / _get_RR_interval(peaks_locs, sampling_rate, beatno, 0),
    "RR_2_1": lambda _0, sampling_rate, peaks_locs, beatno: _get_RR_interval(peaks_locs, sampling_rate, beatno, 1)
    / _get_RR_interval(peaks_locs, sampling_rate, beatno, 0),
    "RR_m_1": lambda _0, sampling_rate, peaks_locs, beatno: _get_mean_RR(peaks_locs, sampling_rate, beatno)
    / _get_RR_interval(peaks_locs, sampling_rate, beatno, 0),
}
# Morphological features from all fiducials
FEATURES_WAVES = {
    "t_PR": lambda sig, sampling_rate, locs_P, _0, locs_R, _1, _2, beatno: _get_diff(
        sig, locs_P, locs_R, sampling_rate, beatno, False
    ),
    "t_QR": lambda sig, sampling_rate, _0, locs_Q, locs_R, _1, _2, beatno: _get_diff(
        sig, locs_Q, locs_R, sampling_rate, beatno, False
    ),
    "t_RS": lambda sig, sampling_rate, _0, _1, locs_R, locs_S, _2, beatno: _get_diff(
        sig, locs_S, locs_R, sampling_rate, beatno, False
    ),
    "t_RT": lambda sig, sampling_rate, _0, _1, locs_R, _2, locs_T, beatno: _get_diff(
        sig, locs_T, locs_R, sampling_rate, beatno, False
    ),
    "t_PQ": lambda sig, sampling_rate, locs_P, locs_Q, _0, _1, _2, beatno: _get_diff(
        sig, locs_P, locs_Q, sampling_rate, beatno, False
    ),
    "t_PS": lambda sig, sampling_rate, locs_P, _0, _1, locs_S, _2, beatno: _get_diff(
        sig, locs_P, locs_S, sampling_rate, beatno, False
    ),
    "t_PT": lambda sig, sampling_rate, locs_P, _0, _1, locs_S, locs_T, beatno: _get_diff(
        sig, locs_P, locs_T, sampling_rate, beatno, False
    ),
    "t_QS": lambda sig, sampling_rate, _0, locs_Q, _1, locs_S, _2, beatno: _get_diff(
        sig, locs_Q, locs_S, sampling_rate, beatno, False
    ),
    "t_QT": lambda sig, sampling_rate, _0, locs_Q, _1, locs_S, locs_T, beatno: _get_diff(
        sig, locs_Q, locs_T, sampling_rate, beatno, False
    ),
    "t_ST": lambda sig, sampling_rate, _0, _1, _2, locs_S, locs_T, beatno: _get_diff(
        sig, locs_S, locs_T, sampling_rate, beatno, False
    ),
    "t_PT_QS": lambda sig, sampling_rate, locs_P, locs_Q, _0, locs_S, locs_T, beatno: _get_diff(
        sig, locs_P, locs_T, sampling_rate, beatno, False
    )
    / _get_diff(sig, locs_Q, locs_S, sampling_rate, beatno, False),
    "t_QT_QS": lambda sig, sampling_rate, _0, locs_Q, _1, locs_S, locs_T, beatno: _get_diff(
        sig, locs_Q, locs_T, sampling_rate, beatno, False
    )
    / _get_diff(sig, locs_Q, locs_S, sampling_rate, beatno, False),
    "a_PQ": lambda sig, sampling_rate, locs_P, locs_Q, _0, _1, _2, beatno: _get_diff(
        sig, locs_P, locs_Q, sampling_rate, beatno, True
    ),
    "a_QR": lambda sig, sampling_rate, _0, locs_Q, locs_R, _1, _2, beatno: _get_diff(
        sig, locs_Q, locs_R, sampling_rate, beatno, True
    ),
    "a_RS": lambda sig, sampling_rate, _0, _1, locs_R, locs_S, _2, beatno: _get_diff(
        sig, locs_R, locs_S, sampling_rate, beatno, True
    ),
    "a_ST": lambda sig, sampling_rate, _0, _1, _2, locs_S, locs_T, beatno: _get_diff(
        sig, locs_S, locs_T, sampling_rate, beatno, True
    ),
    "a_PS": lambda sig, sampling_rate, locs_P, _0, _1, locs_S, _2, beatno: _get_diff(
        sig, locs_P, locs_S, sampling_rate, beatno, True
    ),
    "a_PT": lambda sig, sampling_rate, locs_P, _0, _1, _2, locs_T, beatno: _get_diff(
        sig, locs_P, locs_T, sampling_rate, beatno, True
    ),
    "a_QS": lambda sig, sampling_rate, _0, locs_Q, _1, locs_S, _2, beatno: _get_diff(
        sig, locs_Q, locs_S, sampling_rate, beatno, True
    ),
    "a_QT": lambda sig, sampling_rate, _0, locs_Q, _1, _2, locs_T, beatno: _get_diff(
        sig, locs_Q, locs_T, sampling_rate, beatno, True
    ),
    "a_ST_QS": lambda sig, sampling_rate, _0, locs_Q, _1, locs_S, locs_T, beatno: _get_diff(
        sig, locs_S, locs_T, sampling_rate, beatno, True
    )
    / _get_diff(sig, locs_Q, locs_S, sampling_rate, beatno, True),
    "a_RS_QR": lambda sig, sampling_rate, _0, locs_Q, locs_R, locs_S, _1, beatno: _get_diff(
        sig, locs_R, locs_S, sampling_rate, beatno, True
    )
    / _get_diff(sig, locs_Q, locs_R, sampling_rate, beatno, True),
    "a_PQ_QS": lambda sig, sampling_rate, locs_P, locs_Q, _0, locs_S, _1, beatno: _get_diff(
        sig, locs_P, locs_Q, sampling_rate, beatno, True
    )
    / _get_diff(sig, locs_Q, locs_S, sampling_rate, beatno, True),
    "a_PQ_QT": lambda sig, sampling_rate, locs_P, locs_Q, _0, _1, locs_T, beatno: _get_diff(
        sig, locs_P, locs_Q, sampling_rate, beatno, True
    )
    / _get_diff(sig, locs_Q, locs_T, sampling_rate, beatno, True),
    "a_PQ_PS": lambda sig, sampling_rate, locs_P, locs_Q, _0, locs_S, _1, beatno: _get_diff(
        sig, locs_P, locs_Q, sampling_rate, beatno, True
    )
    / _get_diff(sig, locs_P, locs_S, sampling_rate, beatno, True),
    "a_PQ_QR": lambda sig, sampling_rate, locs_P, locs_Q, locs_R, _0, _1, beatno: _get_diff(
        sig, locs_P, locs_Q, sampling_rate, beatno, True
    )
    / _get_diff(sig, locs_Q, locs_R, sampling_rate, beatno, True),
    "a_PQ_RS": lambda sig, sampling_rate, locs_P, locs_Q, locs_R, locs_S, _0, beatno: _get_diff(
        sig, locs_P, locs_Q, sampling_rate, beatno, True
    )
    / _get_diff(sig, locs_R, locs_S, sampling_rate, beatno, True),
    "a_RS_QS": lambda sig, sampling_rate, _0, locs_Q, locs_R, locs_S, _1, beatno: _get_diff(
        sig, locs_R, locs_S, sampling_rate, beatno, True
    )
    / _get_diff(sig, locs_Q, locs_S, sampling_rate, beatno, True),
    "a_RS_QT": lambda sig, sampling_rate, _0, locs_Q, locs_R, locs_S, locs_T, beatno: _get_diff(
        sig, locs_R, locs_S, sampling_rate, beatno, True
    )
    / _get_diff(sig, locs_Q, locs_T, sampling_rate, beatno, True),
    "a_ST_PQ": lambda sig, sampling_rate, locs_P, locs_Q, _0, locs_S, locs_T, beatno: _get_diff(
        sig, locs_S, locs_T, sampling_rate, beatno, True
    )
    / _get_diff(sig, locs_P, locs_Q, sampling_rate, beatno, True),
    "a_ST_QT": lambda sig, sampling_rate, _0, locs_Q, _1, locs_S, locs_T, beatno: _get_diff(
        sig, locs_S, locs_T, sampling_rate, beatno, True
    )
    / _get_diff(sig, locs_Q, locs_T, sampling_rate, beatno, True),
}


def from_Rpeaks(
    sig: ArrayLike, peaks_locs: ArrayLike, sampling_rate: float, prefix: str = "ecg", average: bool = False
) -> dict:
    """Calculates R-peak-based ECG features and returns a dictionary of features for each heart beat.

        'a_R': Amplitude of R peak
        'RR0': Previous RR interval
        'RR1': Current RR interval
        'RR2': Subsequent RR interval
        'RRm': Mean of RR0, RR1 and RR2
        'RR_0_1': Ratio of RR0 to RR1
        'RR_2_1': Ratio of RR2 to RR1
        'RR_m_1': Ratio of RRm to RR1

    Args:
        sig (ArrayLike): ECG signal segment.
        peaks_locs (ArrayLike): ECG R-peak locations.
        sampling_rate (float): Sampling rate of the ECG signal (Hz).
        prefix (str, optional): Prefix for the feature. Defaults to 'ecg'.
        average (bool, optional): If True, averaged features are returned. Defaults to False.

    Returns:
        dict: Dictionary of ECG features.
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    features_rpeaks = {}
    for m in range(1, len(peaks_locs) - 2):
        features = {}
        for key, func in FEATURES_RPEAKS.items():
            try:
                features["_".join([prefix, key])] = func(sig, sampling_rate, peaks_locs=peaks_locs, beatno=m)
            except:
                features["_".join([prefix, key])] = np.nan
        features_rpeaks[m] = features

    if average:
        features_avr = {}

        features_ = {}
        for subdict in features_rpeaks.values():
            for key, value in subdict.items():
                if key not in features_:
                    features_[key] = [value]
                else:
                    features_[key].append(value)

        for k in features_.keys():
            features_avr[k] = np.mean(features_[k])

        return features_avr

    else:
        return features_rpeaks


def from_waves(
    sig: ArrayLike,
    R_peaks: ArrayLike,
    fiducials: dict,
    sampling_rate: float,
    prefix: str = "ecg",
    average: bool = False,
) -> dict:
    """Calculates ECG features from the given fiducials and returns a dictionary of features.

        't_PR': Time between P and R peak locations
        't_QR': Time between Q and R peak locations
        't_RS': Time between R and S peak locations
        't_RT': Time between R and T peak locations
        't_PQ': Time between P and Q peak locations
        't_PS': Time between P and S peak locations
        't_PT': Time between P and T peak locations
        't_QS': Time between Q and S peak locations
        't_QT':Time between Q and T peak locations
        't_ST': Time between S and T peak locations
        't_PT_QS': Ratio of t_PT to t_QS
        't_QT_QS': Ratio of t_QT to t_QS
        'a_PQ': Difference of P wave and Q wave amplitudes
        'a_QR': Difference of Q wave and R wave amplitudes
        'a_RS': Difference of R wave and S wave amplitudes
        'a_ST': Difference of S wave and T wave amplitudes
        'a_PS': Difference of P wave and S wave amplitudes
        'a_PT': Difference of P wave and T wave amplitudes
        'a_QS': Difference of Q wave and S wave amplitudes
        'a_QT': Difference of Q wave and T wave amplitudes
        'a_ST_QS': Ratio of a_ST to a_QS
        'a_RS_QR': Ratio of a_RS to a_QR
        'a_PQ_QS': Ratio of a_PQ to a_QS
        'a_PQ_QT': Ratio of a_PQ to a_QT
        'a_PQ_PS': Ratio of a_PQ to a_PS
        'a_PQ_QR': Ratio of a_PQ to a_QR
        'a_PQ_RS': Ratio of a_PQ to a_RS
        'a_RS_QS': Ratio of a_RS to a_QS
        'a_RS_QT': Ratio of a_RS to a_QT
        'a_ST_PQ': Ratio of a_ST to a_PQ
        'a_ST_QT': Ratio of a_ST to a_QT

    Args:
        sig (ArrayLike): ECG signal segment.
        R_peaks (ArrayLike): ECG R-peak locations.
        fiducials (dict): Dictionary of fiducial locations (keys: "ECG_P_Peaks", "ECG_Q_Peaks", "ECG_S_Peaks", "ECG_T_Peaks").
        sampling_rate (float): Sampling rate of the ECG signal (Hz).
        prefix (str, optional): Prefix for the feature. Defaults to 'ecg'.
        average (bool, optional): If True, averaged features are returned. Defaults to False.

    Raises:
        ValueError: If sampling rate is not greater than 0.

    Returns:
        dict: Dictionary of ECG features.
    """

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    feature_list = FEATURES_WAVES.copy()

    fiducial_names = ["ECG_P_Peaks", "ECG_Q_Peaks", "ECG_S_Peaks", "ECG_T_Peaks"]
    fiducials = {key: fiducials.get(key, []) for key in fiducial_names}

    P_peaks = fiducials["ECG_P_Peaks"]
    Q_peaks = fiducials["ECG_Q_Peaks"]
    S_peaks = fiducials["ECG_S_Peaks"]
    T_peaks = fiducials["ECG_T_Peaks"]

    if len(P_peaks) == 0:
        P_features = [
            "t_PR",
            "t_PQ",
            "t_PS",
            "t_PT",
            "t_PT_QS",
            "a_PQ",
            "a_PS",
            "a_PT",
            "a_PQ_QS",
            "a_PQ_QT",
            "a_PQ_PS",
            "a_PQ_QR",
            "a_PQ_RS",
            "a_ST_PQ",
        ]
        [feature_list.pop(key, None) for key in P_features]

    if len(Q_peaks) == 0:
        Q_features = [
            "t_QR",
            "t_PQ",
            "t_QS",
            "t_QT",
            "t_PT_QS",
            "t_QT_QS",
            "a_PQ",
            "a_QR",
            "a_QS",
            "a_QT",
            "a_ST_QS",
            "a_RS_QR",
            "a_PQ_QS",
            "a_PQ_QT",
            "a_PQ_PS",
            "a_PQ_QR",
            "a_PQ_RS",
            "a_RS_QS",
            "a_RS_QT",
            "a_ST_PQ",
            "a_ST_QT",
        ]
        [feature_list.pop(key, None) for key in Q_features]

    if len(S_peaks) == 0:
        S_features = [
            "t_SR",
            "t_PS",
            "t_QS",
            "t_ST",
            "t_PT_QS",
            "t_QT_QS",
            "a_RS",
            "a_ST",
            "a_PS",
            "a_QS",
            "a_ST_QS",
            "a_RS_QR",
            "a_PQ_QS",
            "a_PQ_PS",
            "a_PQ_RS",
            "a_RS_QS",
            "a_RS_QT",
            "a_ST_PQ",
            "a_ST_QT",
        ]
        [feature_list.pop(key, None) for key in S_features]

    if len(T_peaks) == 0:
        T_features = [
            "t_TR",
            "t_PT",
            "t_QT",
            "t_ST",
            "t_PT_QS",
            "t_QT_QS",
            "a_ST",
            "a_PT",
            "a_QT",
            "a_ST_QS",
            "a_PQ_QT",
            "a_RS_QT",
            "a_ST_PQ",
            "a_ST_QT",
        ]
        [feature_list.pop(key, None) for key in T_features]

    features_waves = {}
    for m in range(len(R_peaks)):
        features = {}
        for key, func in feature_list.items():
            try:
                features["_".join([prefix, key])] = func(
                    sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno=m
                )
            except:
                features["_".join([prefix, key])] = np.nan
        features_waves[m] = features

    if average:
        features_avr = {}

        features_ = {}
        for subdict in features_waves.values():
            for key, value in subdict.items():
                if key not in features_:
                    features_[key] = [value]
                else:
                    features_[key].append(value)

        for k in features_.keys():
            features_avr[k] = np.mean(features_[k])

        return features_avr

    else:
        return features_waves


def _get_RR_interval(peaks_locs: ArrayLike, sampling_rate: float, beatno: int, interval: int = 0) -> float:

    rr_int = (peaks_locs[beatno + interval + 1] - peaks_locs[beatno + interval]) / sampling_rate

    return rr_int


def _get_mean_RR(peaks_locs: ArrayLike, sampling_rate: float, beatno: int) -> float:

    rr_m = np.mean(
        [
            _get_RR_interval(peaks_locs, sampling_rate, beatno, -1),
            _get_RR_interval(peaks_locs, sampling_rate, beatno, 0),
            _get_RR_interval(peaks_locs, sampling_rate, beatno, 1),
        ]
    )
    return rr_m


def _get_diff(
    sig: ArrayLike,
    loc_array1: ArrayLike,
    loc_array2: ArrayLike,
    sampling_rate: float,
    beatno: int,
    amplitude: bool = False,
) -> float:

    if amplitude:
        feature = sig[loc_array2[beatno]] - sig[loc_array1[beatno]]
    else:
        feature = abs((loc_array2[beatno] - loc_array1[beatno])) / sampling_rate

    return feature
