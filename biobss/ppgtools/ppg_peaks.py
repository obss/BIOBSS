import numpy as np
from numpy.typing import ArrayLike
from biobss.preprocess.signal_detectpeaks import peak_detection

def ppg_beats(sig: ArrayLike , sampling_rate: float, method: str='peakdet', delta: float=None) -> ArrayLike:
    """Detects PPG beats using the 1st derivative of the PPG signal. The detected locations correspond to the rising edge of the PPG beats.

    Args:
        sig (ArrayLike): Signal to be analyzed.
        sampling_rate (float): Sampling rate of the signal (Hz).
        method (str, optional): Peak detection method. Defaults to 'peakdet'.
        delta (float, optional): Delta parameter of the 'peakdet' method. Defaults to None.

    Returns:
        ArrayLike: Beat locations.
    """

    vpg = np.gradient(sig, axis=0, edge_order=1)
    info = peak_detection(vpg, sampling_rate=sampling_rate, method=method, delta=delta)

    return info['Peak_locs']

def peak_control(sig: ArrayLike, peaks_locs: ArrayLike, troughs_locs: ArrayLike, type: str='peak') -> dict:
    """Applies rules to check relative peak and onset locations. 
       First, trims the PPG segment as it starts and ends with a trough.
       Then, checks for missing or duplicate peaks taking the trough lcoations as reference. There must be one peak between successive troughs.

    Args:
        sig (ArrayLike): PPG signal
        peaks_locs (ArrayLike): PPG peak locations
        troughs_locs (ArrayLike): PPG trough locations
        type (str, optional): Type of peaks. It can be 'peak' or 'beat'. Defaults to 'peak'.

    Returns:
        dict: Dictionary of peak locations, peak amplitudes, trough locations and trough amplitudes.
    """
   
    if type == 'beat':
        sig = np.gradient(sig, axis=0, edge_order=1)
    
    peaks_amp = sig[peaks_locs]
    troughs_amp = sig[troughs_locs]

    # Trim the arrays as the signal starts and ends with a trough
    while peaks_locs[0] < troughs_locs[0]:
        peaks_locs = peaks_locs[1:]
        peaks_amp = peaks_amp[1:]

    while peaks_locs[-1] > troughs_locs[-1]:
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
            j = j + np.size(ind_S)

    peaks_locs = loc_S
    peaks_amp = peak_S

    info['Peak_locs'] = peaks_locs
    info['Trough_locs'] = troughs_locs

    return info
