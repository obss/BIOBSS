import numpy as np
from numpy.typing import ArrayLike
from biobss.preprocess.signal_detectpeaks import peak_detection

def ppg_beats(sig: ArrayLike , sampling_rate: float, method: str='peakdet', delta: float=None) -> ArrayLike:

    vpg = np.gradient(sig, axis=0, edge_order=1)
    info = peak_detection(vpg, sampling_rate=sampling_rate, method=method, delta=delta)

    return info['Peak_locs']

def peak_control(peaks_locs: ArrayLike, peaks_amp: ArrayLike, troughs_locs: ArrayLike, troughs_amp: ArrayLike) -> dict:
    """Applies rules to check relative peak and onset locations. 
       First, trims the PPG segment as it starts and ends with a trough.
       Then, checks for missing or duplicate peaks taking the trough lcoations as reference. There must be one peak between successive troughs.

    Args:
        peaks_locs (array): PPG peak locations
        peaks_amp (array): PPG peak amplitudes
        troughs_locs (array): PPG trough locations
        troughs_amp (array): PPG trough amplitudes

    Returns:
        info(dict): Dictionary of peak locations, peak amplitudes, trough locations and trough amplitudes.
    """
    if (len(peaks_locs) != len(peaks_amp)):
        raise ValueError("Lengths of peak location and peak amplitude arrays do not match!")
    if (len(troughs_locs) != len(troughs_amp)):
        raise ValueError("Lengths of trough location and trough amplitude arrays do not match!")

    # Trim the arrays as the signal starts and ends with a trough
    if peaks_locs[0] < troughs_locs[0]:
        peaks_locs = peaks_locs[1:]
        peaks_amp = peaks_amp[1:]

    if peaks_locs[-1] > troughs_locs[-1]:
        peaks_locs = peaks_locs[:-1]
        peaks_amp = peaks_amp[:-1]

    # Apply rules to check if there are missing or duplicate peaks
    info = {}

    search_S = troughs_locs
    loc_S = []
    peak_S = []
    j = 0

    for i in range(len(search_S)-1):

        ind_S = np.asarray(
            np.where((search_S[i] < peaks_locs) & (peaks_locs < search_S[i+1])))

        if np.size(ind_S) == 0:
            peak_S.insert(i, np.NaN)
            loc_S.insert(i, np.NaN)
            j = j+1

        elif np.size(ind_S) == 1:
            peak_S.insert(i, peaks_amp[j])
            loc_S.insert(i, peaks_locs[j])
            j = j+1

        else:
            peak_mx = np.max(peaks_amp[ind_S])
            ind_mx = np.argmax(peaks_amp[ind_S])
            peak_S.insert(i, peak_mx)
            loc_S.insert(i, peaks_locs[ind_S[ind_mx][0]])
            j = j+1

    peaks_locs = loc_S
    peaks_amp = peak_S

    info['Peak_locs'] = peaks_locs
    info['Peaks'] = peaks_amp
    info['Trough_locs'] = troughs_locs
    info['Troughs'] = troughs_amp

    return info
