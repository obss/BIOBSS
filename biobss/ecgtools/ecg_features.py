import numpy as np
from numpy.typing import ArrayLike

#Morphological features from R-peak locations
FEATURES_RPEAKS = {
'a_R': lambda sig, _0, peaks_locs,  beatno: sig[peaks_locs[beatno]],
'RR0': lambda _0, sampling_rate, peaks_locs, beatno: _get_RR_interval(peaks_locs, sampling_rate, beatno, -1),
'RR1': lambda _0, sampling_rate, peaks_locs, beatno: _get_RR_interval(peaks_locs, sampling_rate, beatno, 0),
'RR2': lambda _0, sampling_rate, peaks_locs, beatno: _get_RR_interval(peaks_locs, sampling_rate, beatno, 1),
'RRm': lambda _0, sampling_rate, peaks_locs, beatno: _get_mean_RR(peaks_locs, sampling_rate, beatno),
'RR_0_1': lambda _0, sampling_rate, peaks_locs, beatno: _get_RR_interval(peaks_locs, sampling_rate, beatno, -1) / _get_RR_interval(peaks_locs, sampling_rate, beatno, 0),
'RR_2_1': lambda _0, sampling_rate, peaks_locs, beatno: _get_RR_interval(peaks_locs, sampling_rate, beatno, 1) / _get_RR_interval(peaks_locs, sampling_rate, beatno, 0),
'RR_m_1': lambda _0, sampling_rate, peaks_locs, beatno: _get_mean_RR(peaks_locs, sampling_rate, beatno) / _get_RR_interval(peaks_locs, sampling_rate, beatno, 0),
}

FEATURES_WAVES = {
't_PR': lambda sig, sampling_rate, locs_P, _0, locs_R, _1, _2, beatno: _get_diff(sig, locs_P, locs_R, sampling_rate, beatno, False),
't_QR': lambda sig, sampling_rate, _0, locs_Q, locs_R, _1, _2, beatno: _get_diff(sig, locs_Q, locs_R, sampling_rate, beatno, False), 
't_SR': lambda sig, sampling_rate, _0, _1, locs_R, locs_S, _2, beatno: _get_diff(sig, locs_S, locs_R, sampling_rate, beatno, False),
't_TR': lambda sig, sampling_rate, _0, _1, locs_R, _2, locs_T, beatno: _get_diff(sig, locs_T, locs_R, sampling_rate, beatno, False),
't_PQ': lambda sig, sampling_rate, locs_P, locs_Q, _0, _1, _2, beatno: _get_diff(sig, locs_P, locs_Q, sampling_rate, beatno, False),
't_PS': lambda sig, sampling_rate, locs_P, _0, _1, locs_S, _2, beatno: _get_diff(sig, locs_P, locs_S, sampling_rate, beatno, False),
't_PT': lambda sig, sampling_rate, locs_P, _0, _1, locs_S, locs_T, beatno: _get_diff(sig, locs_P, locs_T, sampling_rate, beatno, False),
't_QS': lambda sig, sampling_rate, _0, locs_Q, _1, locs_S, _2, beatno: _get_diff(sig, locs_Q, locs_S, sampling_rate, beatno, False),
't_QT': lambda sig, sampling_rate, _0, locs_Q, _1, locs_S, locs_T, beatno: _get_diff(sig, locs_Q, locs_T, sampling_rate, beatno, False),
't_ST': lambda sig, sampling_rate, _0, _1, _2, locs_S, locs_T, beatno: _get_diff(sig, locs_S, locs_T, sampling_rate, beatno, False),
't_PT_QS': lambda sig, sampling_rate, locs_P, locs_Q, _0, locs_S, locs_T, beatno: _get_diff(sig, locs_P, locs_T, sampling_rate, beatno, False)/_get_diff(sig, locs_Q, locs_S, sampling_rate, beatno, False), 
't_QT_QS': lambda sig, sampling_rate, _0, locs_Q, _1, locs_S, locs_T, beatno: _get_diff(sig, locs_Q, locs_T, sampling_rate, beatno, False)/_get_diff(sig, locs_Q, locs_S, sampling_rate, beatno, False),
'a_PQ': lambda sig, sampling_rate, locs_P, locs_Q, _0, _1, _2, beatno: _get_diff(sig, locs_P, locs_Q, sampling_rate, beatno, True),
'a_QR': lambda sig, sampling_rate, _0, locs_Q, locs_R, _1, _2, beatno: _get_diff(sig, locs_Q, locs_R, sampling_rate, beatno, True),
'a_RS': lambda sig, sampling_rate, _0, _1, locs_R, locs_S, _2, beatno: _get_diff(sig, locs_R, locs_S, sampling_rate, beatno, True),
'a_ST': lambda sig, sampling_rate, _0, _1, _2, locs_S, locs_T, beatno: _get_diff(sig, locs_S, locs_T, sampling_rate, beatno, True),
'a_PS': lambda sig, sampling_rate, locs_P, _0, _1, locs_S, _2, beatno: _get_diff(sig, locs_P, locs_S, sampling_rate, beatno, True),
'a_PT': lambda sig, sampling_rate, locs_P, _0, _1, _2, locs_T, beatno: _get_diff(sig, locs_P, locs_T, sampling_rate, beatno, True),
'a_QS': lambda sig, sampling_rate, _0, locs_Q, _1, locs_S, _2, beatno: _get_diff(sig, locs_Q, locs_S, sampling_rate, beatno, True),
'a_QT': lambda sig, sampling_rate, _0, locs_Q, _1, _2, locs_T, beatno: _get_diff(sig, locs_Q, locs_T, sampling_rate, beatno, True),
'a_ST_QS': lambda sig, sampling_rate, _0, locs_Q, _1, locs_S, locs_T, beatno: _get_diff(sig, locs_S, locs_T, sampling_rate, beatno, True)/_get_diff(sig, locs_Q, locs_S, sampling_rate, beatno, True),
'a_RS_QR': lambda sig, sampling_rate, _0, locs_Q, locs_R, locs_S, _1, beatno: _get_diff(sig, locs_R, locs_S, sampling_rate, beatno, True)/_get_diff(sig, locs_Q, locs_R, sampling_rate, beatno, True),
'a_PQ_QS': lambda sig, sampling_rate, locs_P, locs_Q, _0, locs_S, _1, beatno: _get_diff(sig, locs_P, locs_Q, sampling_rate, beatno, True)/_get_diff(sig, locs_Q, locs_S, sampling_rate, beatno, True),
'a_PQ_QT': lambda sig, sampling_rate, locs_P, locs_Q, _0, _1, locs_T, beatno: _get_diff(sig, locs_P, locs_Q, sampling_rate, beatno, True)/_get_diff(sig, locs_Q, locs_T, sampling_rate, beatno, True),
'a_PQ_PS': lambda sig, sampling_rate, locs_P, locs_Q, _0, locs_S, _1, beatno: _get_diff(sig, locs_P, locs_Q, sampling_rate, beatno, True)/_get_diff(sig, locs_P, locs_S, sampling_rate, beatno, True),
'a_PQ_QR': lambda sig, sampling_rate, locs_P, locs_Q, locs_R, _0, _1, beatno: _get_diff(sig, locs_P, locs_Q, sampling_rate, beatno, True)/_get_diff(sig, locs_Q, locs_R, sampling_rate, beatno, True),
'a_PQ_RS': lambda sig, sampling_rate, locs_P, locs_Q, locs_R, locs_S, _0, beatno: _get_diff(sig, locs_P, locs_Q, sampling_rate, beatno, True)/_get_diff(sig, locs_R, locs_S, sampling_rate, beatno, True),
'a_RS_QS': lambda sig, sampling_rate, _0, locs_Q, locs_R, locs_S, _1, beatno: _get_diff(sig, locs_R, locs_S, sampling_rate, beatno, True)/_get_diff(sig, locs_Q, locs_S, sampling_rate, beatno, True),
'a_RS_QT': lambda sig, sampling_rate, _0, locs_Q, locs_R, locs_S, locs_T, beatno: _get_diff(sig, locs_R, locs_S, sampling_rate, beatno, True)/_get_diff(sig, locs_Q, locs_T, sampling_rate, beatno, True),
'a_ST_PQ': lambda sig, sampling_rate, locs_P, locs_Q, _0, locs_S, locs_T, beatno: _get_diff(sig, locs_S, locs_T, sampling_rate, beatno, True)/_get_diff(sig, locs_P, locs_Q, sampling_rate, beatno, True),
'a_ST_QT': lambda sig, sampling_rate, _0, locs_Q, _1, locs_S, locs_T, beatno: _get_diff(sig, locs_S, locs_T, sampling_rate, beatno, True)/_get_diff(sig, locs_Q, locs_T, sampling_rate, beatno, True),

}

def from_Rpeaks(sig: ArrayLike, peaks_locs: ArrayLike, sampling_rate: float, prefix: str='ecg') -> dict:
    """Calculates R-peak-based ECG features and returns a dictionary of features.

    Args:
        sig (ArrayLike): ECG signal segment to be analyzed
        peaks_locs (ArrayLike): ECG R-peak locations
        peaks_amp (ArrayLike): ECG R-peak amplitudes
        sampling_rate (float): Sampling rate of the ECG signal.
        prefix (str, optional): Prefix for signal type. Defaults to 'ecg'.

    Returns:
        dict: Dictionary of calculated features.
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    features_rpeaks={}
    for m in range(len(peaks_locs)):
        features={}
        for key,func in FEATURES_RPEAKS.items():
            features["_".join([prefix, key])]=func(sig, sampling_rate, peaks_locs=peaks_locs, beatno=m)
        features_rpeaks[m] = features
        
    return features_rpeaks

def from_waves(sig: ArrayLike, R_peaks: ArrayLike, fiducials: dict, sampling_rate: float, prefix: str='ecg') -> dict:
    """Calculates ECG features from all fiducials and returns a dictionary of features.

    Args:
        sig (ArrayLike): ECG signal segment to be analyzed
        fiducials (dict): Dictionary of fiducial locations (keys: "ECG_P_Peaks", "ECG_Q_Peaks", "ECG_S_Peaks", "ECG_T_Peaks", "ECG_P_Onsets", "ECG_T_Offsets")
        sampling_rate (float): Sampling rate of the ECG signal
        prefix (str, optional): Prefix for signal type. Defaults to 'ecg'.

    Raises:
        ValueError: If sampling rate is not greater than 0. 

    Returns:
        dict: Dictionary of calculated features.
    """

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    P_peaks = fiducials['ECG_P_Peaks']
    Q_peaks = fiducials['ECG_Q_Peaks']
    S_peaks = fiducials['ECG_S_Peaks']
    T_peaks = fiducials['ECG_T_Peaks']
    P_onsets = fiducials['ECG_P_Onsets']
    T_offsets = fiducials['ECG_T_Offsets']

    features_waves={}
    for m in range(len(R_peaks)):
        features={}
        for key,func in FEATURES_WAVES.items():
            features["_".join([prefix, key])]=func(sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno=m)
        features_waves['m'] = features
  
    return features_waves


def _get_RR_interval(peaks_locs, sampling_rate, beatno, interval=0):

    rr_int = (peaks_locs[beatno - 1 + interval] - peaks_locs[beatno - 2 + interval]) / sampling_rate

    return rr_int

def _get_mean_RR(peaks_locs, sampling_rate, beatno):

    rr_m = np.mean([_get_RR_interval(peaks_locs, sampling_rate, beatno, -1), _get_RR_interval(peaks_locs, sampling_rate, beatno, 0), _get_RR_interval(peaks_locs, sampling_rate, beatno, 1)])
    return rr_m

def _get_diff(sig, loc_array1, loc_array2, sampling_rate, beatno, amplitude=False):

    if not amplitude:
        feature = abs((loc_array2[beatno] -loc_array1[beatno]))/sampling_rate

    else:
        feature = (sig[loc_array2[beatno]] -sig[loc_array1[beatno]])

    return feature