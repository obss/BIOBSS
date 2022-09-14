import numpy as np
from numpy.typing import ArrayLike


def peak_control(locs_peaks: ArrayLike, peaks: ArrayLike, locs_troughs: ArrayLike, troughs: ArrayLike) -> dict:
    """Applies rules to check relative peak and onset locations. 
       First, trims the PPG segment as it starts and ends with a trough.
       Then, checks for missing or duplicate peaks taking the trough lcoations as reference. There must be one peak between successive troughs.

    Args:
        locs_peaks (array): PPG peak locations
        peaks (array): PPG peak amplitudes
        locs_troughs (array): PPG trough locations
        troughs (array): PPG trough amplitudes

    Returns:
        info(dict): Dictionary of peak locations, peak amplitudes, trough locations and trough amplitudes.
    """

    # Trim the arrays as the signal starts and ends with a trough

    if locs_peaks[0] < locs_troughs[0]:
        locs_peaks = locs_peaks[1:]
        peaks = peaks[1:]

    if locs_peaks[-1] > locs_troughs[-1]:
        locs_peaks = locs_peaks[:-1]
        peaks = peaks[:-1]

    # Apply rules to check if there are missing or duplicate peaks
    info = {}

    search_S = locs_troughs
    loc_S = []
    peak_S = []
    j = 0

    for i in range(len(search_S)-1):

        ind_S = np.asarray(
            np.where((search_S[i] < locs_peaks) & (locs_peaks < search_S[i+1])))

        if np.size(ind_S) == 0:

            peak_S.insert(i, np.NaN)
            loc_S.insert(i, np.NaN)
            j = j+1
        elif np.size(ind_S) == 1:

            peak_S.insert(i, peaks[j])
            loc_S.insert(i, locs_peaks[j])
            j = j+1
        else:

            peak_mx = np.max(peaks[ind_S])
            ind_mx = np.argmax(peaks[ind_S])
            peak_S.insert(i, peak_mx)
            loc_S.insert(i, locs_peaks[ind_S[ind_mx][0]])
            j = j+1

    locs_peaks = loc_S
    peaks = peak_S

    info['Peak_locs'] = locs_peaks
    info['Peaks'] = peaks
    info['Trough_locs'] = locs_troughs
    info['Troughs'] = troughs

    return info
