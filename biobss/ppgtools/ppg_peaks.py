import numpy as np
from numpy.typing import ArrayLike

from biobss.preprocess.signal_detectpeaks import peak_detection


def ppg_detectpeaks(
    sig: ArrayLike,
    sampling_rate: float,
    method: str = "peakdet",
    correct_peaks: bool = True,
    delta: float = None,
    type: str = "peak",
) -> dict:
    """Detects peaks and troughs of PPG signal.

    Args:
        sig (ArrayLike): PPG signal.
        sampling_rate (float): Sampling rate of the PPG signal (Hz).
        method (str, optional): Peak detection method. Should be one of 'peakdet', 'heartpy' and 'scipy'. Defaults to 'peakdet'.
                                See https://gist.github.com/endolith/250860 to get information about 'peakdet' method.
        correct_peaks (bool, optional): If True, peak locations are corrected relative to trough locations.  Defaults to True.
        delta (float, optional): Delta parameter of the 'peakdet' method. Defaults to None.
        type (str, optional): Type of peaks. It can be 'peak' or 'beat'. Defaults to 'peak'.

    Raises:
        ValueError: If sampling rate is not greater than 0.

    Returns:
        dict: Dictionary of peak and trough locations.
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    info = peak_detection(sig=sig, sampling_rate=sampling_rate, method=method, delta=delta)
    locs_peaks = info["Peak_locs"]
    locs_troughs = info["Trough_locs"]

    if type == "beat":
        locs_peaks = ppg_detectbeats(sig=sig, sampling_rate=sampling_rate, method=method, delta=delta)

    if correct_peaks:
        info = peak_control(sig=sig, peaks_locs=locs_peaks, troughs_locs=locs_troughs, type=type)

    return info


def ppg_detectbeats(sig: ArrayLike, sampling_rate: float, method: str = "peakdet", delta: float = None) -> ArrayLike:
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

    return info["Peak_locs"]


def peak_control(sig: ArrayLike, peaks_locs: ArrayLike, troughs_locs: ArrayLike, type: str = "peak") -> dict:
    """Applies rules to check relative peak and onset locations.
       First, trims the PPG segment as it starts and ends with a trough.
       Then, checks for missing or duplicate peaks taking the trough lcoations as reference. There must be one peak between successive troughs.

    Args:
        sig (ArrayLike): PPG signal
        peaks_locs (ArrayLike): PPG peak locations
        troughs_locs (ArrayLike): PPG trough locations
        type (str, optional): Type of peaks. It can be 'peak' or 'beat'. Defaults to 'peak'.

    Returns:
        dict: Dictionary of peak and trough locations.
    """

    if type == "beat":
        sig = np.gradient(sig, axis=0, edge_order=1)

    peaks_locs = np.asarray(peaks_locs)
    troughs_locs = np.asarray(troughs_locs)

    peaks_amp = sig[peaks_locs]

    # Trim the arrays as the signal starts and ends with a trough
    while peaks_locs[0] < troughs_locs[0]:
        peaks_locs = peaks_locs[1:]
        peaks_amp = peaks_amp[1:]

    while peaks_locs[-1] > troughs_locs[-1]:
        peaks_locs = peaks_locs[:-1]
        peaks_amp = peaks_amp[:-1]

    # Apply rules to check if there are missing or duplicate peaks
    # find_missing_duplicate_peaks(locs_valleys=troughs_locs, locs_peaks=peaks_locs, peaks=peaks_amp)

    info = {}

    locs_, amps_ = correct_missing_duplicate_peaks(locs_valleys=troughs_locs, locs_peaks=peaks_locs, peaks=peaks_amp)

    info["Peak_locs"] = locs_
    info["Trough_locs"] = troughs_locs

    return info


def ppg_waves(
    sig: ArrayLike,
    locs_onsets: ArrayLike,
    sampling_rate: float,
    th_w: float = 0.5,
    th_y: float = 0.45,
    th_a: float = 0.45,
) -> dict:
    """Detects fiducials of PPG, VPG and APG signals.

    Args:
        sig (ArrayLike): PPG signal.
        locs_onsets (ArrayLike): PPG signal onset locations
        sampling_rate (float): Sampling rate of the PPG signal (Hz).
        th_w (float, optional): Threshold to detect w waves. Defaults to 0.5.
        th_y (float, optional): Threshold to detect y waves. Defaults to 0.45.
        th_a (float, optional): Threshold to detect a waves. Defaults to 0.45.

    Returns:
        dict: Dictionary of fiducial locations.
    """
    vpg_sig = np.gradient(sig) / (1 / sampling_rate)
    apg_sig = np.gradient(vpg_sig) / (1 / sampling_rate)

    fiducials = {}

    vpg_fiducials = vpg_delineate(vpg_sig=vpg_sig, sampling_rate=sampling_rate, th_w=th_w, th_y=th_y)
    apg_fiducials = apg_delineate(
        apg_sig=apg_sig, vpg_sig=vpg_sig, vpg_fiducials=vpg_fiducials, sampling_rate=sampling_rate, th_a=th_a, th_w=th_w
    )
    ppg_fiducials = ppg_delineate(
        ppg_sig=sig,
        vpg_sig=vpg_sig,
        vpg_fiducials=vpg_fiducials,
        apg_sig=apg_sig,
        apg_fiducials=apg_fiducials,
        sampling_rate=sampling_rate,
        locs_onsets=locs_onsets,
    )

    fiducials.update(ppg_fiducials)
    fiducials.update(vpg_fiducials)
    fiducials.update(apg_fiducials)

    return fiducials


def vpg_delineate(vpg_sig: ArrayLike, sampling_rate: float, th_w: float = 0.5, th_y: float = 0.45) -> dict:
    """Detects fiducials of VPG signal.

        This function is an implementation of the method presented in:
        Abhishek Chakraborty, Deboleena Sadhukhan & Madhuchhanda Mitra
        (2019): An Automated Algorithm to Extract Time Plane Features From the PPG Signal and
        its Derivatives for Personal Health Monitoring Application, IETE Journal of Research, DOI:
        10.1080/03772063.2019.1604178

    Args:
        vpg_sig (ArrayLike): VPG signal.
        sampling_rate (float): Sampling rate of the VPG signal (Hz).
        th_w (float, optional): Threshold to detect w waves. Defaults to 0.5.
        th_y (float, optional): Threshold to detect y waves. Defaults to 0.45.

    Returns:
        dict: Dictionary of fiducial locations.
    """
    fiducials = {}

    ###########################################
    # Detect w-waves (maximum peak of VPG signal)
    ###########################################

    max_amp = np.max(vpg_sig)
    thr_w = th_w * max_amp

    cropped_vpg = vpg_sig.copy()
    cropped_vpg[vpg_sig < thr_w] = 0

    w_len = round(0.25 * sampling_rate)

    # Generate the search indices
    ind = _generate_search_indices(w_len=w_len, sig_len=len(cropped_vpg))

    # Search for a slope reversal point in each window (w_len)
    locs_w = _search_slope_reversals(
        cropped_vpg, direction="negative", search_direction="left_to_right", criterion="all", search_indices=ind
    )

    ###########################################
    # Detect y-waves (minimum trough of VPG signal)
    ###########################################

    min_amp = min(vpg_sig)
    thr_y = th_y * min_amp

    # Define a search interval
    search_st = locs_w
    locs = np.append(locs_w, len(vpg_sig))
    search_int = np.round(np.diff(locs) / 2).astype(int)
    search_end = search_st + search_int
    cropped_vpg = np.zeros(len(vpg_sig))

    for k in range(len(search_int)):
        cropped_vpg[search_st[k] : search_end[k]] = vpg_sig[search_st[k] : search_end[k]]

    cropped_vpg[vpg_sig > thr_y] = 0

    # Generate the search indices
    ind = _generate_search_indices(w_len=w_len, sig_len=len(cropped_vpg))

    # Search for a slope reversal in each window (w_len)
    search_st = locs_w
    search_e = np.append(locs_w[1:], len(vpg_sig) - 1)
    locs_y = _search_slope_reversals(
        sig=cropped_vpg,
        direction="positive",
        search_direction="left_to_right",
        criterion="min",
        search_start=search_st,
        search_end=search_e,
    )
    peaks_y = vpg_sig[locs_y]

    ###########################################
    # Detect z-waves (local extreme)
    ###########################################

    # search for the next zero crossing point
    search_st = locs_y + 1
    search_end = np.append(search_st[1:], len(vpg_sig))
    zero_crossings = _search_zero_crossings(
        sig=vpg_sig,
        search_start=search_st,
        search_end=search_end,
        direction="positive",
        search_direction="left_to_right",
        criterion="first",
    )
    zero_cross = np.array([x[0] for x in zero_crossings])

    # if number of y peaks and zero crossing points are not equal
    if len(zero_cross) != len(locs_y):
        if locs_y[0] > zero_cross[0]:
            zero_cross = np.insert(zero_cross, 0, np.mean(np.diff(zero_cross)))

        if locs_y[-1] > zero_cross[-1]:
            zero_cross = np.append(zero_cross, len(vpg_sig) - 1)

    # modify the signal (first 70% percent of the samples)
    nofsamp = np.round(0.7 * (zero_cross - locs_y)).astype(int)

    a1 = locs_y
    b1 = peaks_y
    a2 = locs_y + nofsamp
    b2 = vpg_sig[a2]

    m = (b2 - b1) / (a2 - a1)

    mod_samp = vpg_sig.copy()

    max_ind = []
    for j in range(len(m)):
        i = 0
        for k in range(a1[j], a2[j] + 1):
            mod_samp[k] = vpg_sig[k] - m[j] * i - b1[j]
            i += 1

        # mod_max = np.max(mod_samp[a1[j]:a2[j]+1])  # Find the maximum
        ind = np.argmax(mod_samp[a1[j] : a2[j] + 1])  # Find the index of maximum
        ind += a1[j]
        max_ind.append(ind)

    locs_z = np.array(max_ind)

    fiducials["w_waves"] = locs_w
    fiducials["y_waves"] = locs_y
    fiducials["z_waves"] = locs_z

    return fiducials


def apg_delineate(
    apg_sig: ArrayLike,
    vpg_sig: ArrayLike,
    vpg_fiducials: dict,
    sampling_rate: float,
    th_a: float = 0.45,
    th_w: float = 0.5,
) -> dict:
    """Detects fiducials of APG signal.

        This function is an implementation of the method presented in:
        Abhishek Chakraborty, Deboleena Sadhukhan & Madhuchhanda Mitra
        (2019): An Automated Algorithm to Extract Time Plane Features From the PPG Signal and
        its Derivatives for Personal Health Monitoring Application, IETE Journal of Research, DOI:
        10.1080/03772063.2019.1604178

    Args:
        apg_sig (ArrayLike): APG signal.
        vpg_sig (ArrayLike): VPG signal.
        vpg_fiducials (dict): VPG fiducials.
        sampling_rate (float): Sampling rate of the APG signal (Hz).
        th_a (float, optional): Threshold to detect a waves. Defaults to 0.45.
        th_w (float, optional): Threshold to detect w waves. Defaults to 0.5.

    Returns:
        dict: _description_
    """

    locs_y = vpg_fiducials["y_waves"]
    locs_z = vpg_fiducials["z_waves"]

    fiducials = {}

    ###########################################
    # Detect a-waves (maximum peak of APG signal)
    ###########################################

    max_amp = np.max(apg_sig)
    thr_a = th_a * max_amp

    cropped_apg = apg_sig.copy()
    cropped_apg[apg_sig < thr_a] = 0

    w_len = round(0.25 * sampling_rate)

    thr_w = th_w * np.max(vpg_sig)
    cropped_vpg = vpg_sig.copy()
    cropped_vpg[vpg_sig < thr_w] = 0

    # Generate the search indices
    ind = _generate_search_indices(w_len, sig_len=len(cropped_vpg))

    # Search for a slope reversal point in each segment (w_len)

    # it should be 'negative' and 'left_to_right' but the equality checks match better for this combination
    # check the paper and consider refactoring _find_slope_reversals
    locs_a = _search_slope_reversals(
        sig=cropped_apg, search_indices=ind, direction="positive", search_direction="right_to_left", criterion="all"
    )

    ###########################################
    # Detect b-waves (minimum trough of APG signal)
    ###########################################

    # Search for the next zero-crossing point
    search_st = locs_a + 1
    search_end = np.append(search_st[1:], len(apg_sig))
    zero_crossings = _search_zero_crossings(
        sig=apg_sig,
        search_start=search_st,
        search_end=search_end,
        direction="negative",
        search_direction="left_to_right",
        criterion="first",
    )

    zero_cross = np.array([x[0] for x in zero_crossings])

    # search for the first slope reversal point
    search_st = zero_cross
    search_e = np.append(zero_cross[1:], len(apg_sig) - 1)
    locs_b = _search_slope_reversals(
        sig=apg_sig,
        direction="positive",
        search_direction="left_to_right",
        criterion="first",
        search_start=search_st,
        search_end=search_e,
    )
    peaks_b = apg_sig[locs_b]

    ###########################################
    # Detect e-waves
    ###########################################

    # Generate search indices
    ind = []
    for k in range(len(locs_z)):
        ind.append(np.arange(locs_y[k] - 5, locs_z[k] + 6))

    # Search for a slope reversal point (from z to y, on APG)
    locs_e = _search_slope_reversals(
        sig=apg_sig, direction="positive", search_direction="right_to_left", criterion="max", search_indices=ind
    )
    peaks_e = apg_sig[locs_e]

    ###########################################
    # Detect c and d-waves
    ###########################################

    # 1. Search for slope reversal points (from e to b)
    # First: d, second: c

    locs_c = []
    peaks_c = []
    locs_d = []
    peaks_d = []

    for h in range(len(locs_e)):
        seg = apg_sig[locs_b[h] + 1 : locs_e[h] - 1]
        slope = np.diff(seg)

        p = len(slope)
        locs = []
        peaks = []

        while p > 1:
            if (slope[p - 1] < 0 and slope[p - 2] >= 0) or (slope[p - 1] >= 0 and slope[p - 2] < 0):
                loc = locs_b[h] + p - 1
                peak = apg_sig[loc]
                peaks.append(peak)
                locs.append(loc)

            p -= 1

        if len(locs) >= 2:
            locs_d.append(locs[0])
            peaks_d.append(peaks[0])

            locs_c.append(locs[1])
            peaks_c.append(peaks[1])

        else:
            # If no slope reversal point is found on
            # the SDPPG data, then starting from the location of b peak, all the
            # samples up to the location of e peak are modified (Equation 4).

            a1 = locs_b[h]
            b1 = peaks_b[h]
            a2 = locs_e[h]
            b2 = peaks_e[h]

            m = (b2 - b1) / (a2 - a1)
            mod_samp = apg_sig[a1:a2]

            mod_samp = np.empty(len(range(a1, a2)))

            for i, k in enumerate(range(a1, a2)):
                mod_samp[i] = apg_sig[k] - m * i

            # mod_max=np.max(mod_samp)
            ind = np.argmax(mod_samp)
            max_ind = ind + a1

            # Now, among these modified APG signals, the maximum peak location
            # is determined at first. Then takin reference to the location of
            # this maximum peak, a right traversal is carried out on those
            # modified APG samples to find out the first slope change point as
            # the location of c peak and the next slope change point as the
            # location of d peak.T

            seg = mod_samp[ind:]
            slope = np.diff(seg)

            locs2 = []
            peaks2 = []

            for s in range(len(slope) - 1):
                if (slope[s] < 0 and slope[s + 1] >= 0) or (slope[s] >= 0 and slope[s + 1] < 0):
                    loc = max_ind + s
                    peak = apg_sig[loc]
                    peaks2.append(peak)
                    locs2.append(loc)

            if len(locs2) >= 2:
                locs_c.append(locs2[0])
                peaks_c.append(peaks2[0])

                locs_d.append(locs2[1])
                peaks_d.append(peaks2[1])

            else:
                # If both of the above-mentioned logic fails, the
                # algorithm will consider overlapping c, d and e
                # waves.
                locs_c.append(locs_e[h])
                peaks_c.append(peaks_e[h])

                locs_d.append(locs_e[h])
                peaks_d.append(peaks_e[h])

    fiducials["a_waves"] = np.asarray(locs_a)
    fiducials["b_waves"] = np.asarray(locs_b)
    fiducials["e_waves"] = np.asarray(locs_e)
    fiducials["c_waves"] = np.asarray(locs_c)
    fiducials["d_waves"] = np.asarray(locs_d)

    return fiducials


def ppg_delineate(
    ppg_sig: ArrayLike,
    vpg_sig: ArrayLike,
    vpg_fiducials: dict,
    apg_sig: ArrayLike,
    apg_fiducials: dict,
    sampling_rate: float,
    locs_onsets: ArrayLike = None,
) -> dict:
    """Detects fiducials of PPG signal.

        This function is an implementation of the method presented in:
        Abhishek Chakraborty, Deboleena Sadhukhan & Madhuchhanda Mitra
        (2019): An Automated Algorithm to Extract Time Plane Features From the PPG Signal and
        its Derivatives for Personal Health Monitoring Application, IETE Journal of Research, DOI:
        10.1080/03772063.2019.1604178

    Args:
        ppg_sig (ArrayLike): PPG signal.
        vpg_sig (ArrayLike): VPG signal.
        vpg_fiducials (dict): VPG fiducials.
        apg_sig (ArrayLike): APG signal.
        apg_fiducials (dict): APG fiducials.
        sampling_rate (float): Sampling rate of the PPG signal (Hz).
        locs_onsets (ArrayLike, optional): PPG signal onset locations. Defaults to None.

    Returns:
        dict: Dictionary of fiducial locations
    """
    locs_w = vpg_fiducials["w_waves"]
    locs_e = apg_fiducials["e_waves"]

    fiducials = {}

    ###########################################
    # Detect O-waves (pulse onsets)
    ###########################################

    if locs_onsets is None:
        info = peak_detection(ppg_sig, sampling_rate, "peakdet", delta=0.01)
        locs_O = info["Trough_locs"]
        locs_peaks = info["Peak_locs"]

        info2 = peak_control(ppg_sig, peaks_locs=locs_peaks, troughs_locs=locs_O, type="peak")
        locs_O = info2["Trough_locs"]
        locs_peaks = info2["Peak_locs"]

    else:
        locs_O = locs_onsets

    ###########################################
    # Detect S-waves (systolic peaks)
    ###########################################

    # Search for the first zero crossing poing (from w to the right)

    search_st = locs_w + 1
    search_e = np.append(search_st[1:], len(vpg_sig) - 1)
    zero_crossings = _search_zero_crossings(
        sig=vpg_sig,
        direction="negative",
        search_direction="left_to_right",
        criterion="first",
        search_start=search_st,
        search_end=search_e,
    )
    zero_cross = np.array([x[0] for x in zero_crossings])

    # Select twenty samples of the PPG signal around each of zero crossing point.

    search_st = [x - 10 for x in zero_cross]
    search_e = [x + 10 for x in zero_cross]

    # Then, find the slope reversal point among these marked samples

    locs_S = _search_slope_reversals(
        sig=ppg_sig,
        direction="negative",
        search_direction="left_to_right",
        criterion="first",
        search_start=search_st,
        search_end=search_e,
    )

    ###########################################
    # Detect N-waves (dicrotic notches)
    ###########################################

    locs_N = locs_e

    ###########################################
    # Detect D-waves (diastolic peaks)
    ###########################################

    # Search for a slope reversal point (from e to the right)
    search_st = locs_e + 1
    search_e = np.append(locs_e[1:] - 1, len(apg_sig) - 1)

    locs_D = _search_slope_reversals(
        sig=apg_sig,
        direction="positive",
        search_direction="left_to_right",
        criterion="first",
        search_start=search_st,
        search_end=search_e,
    )

    fiducials["O_waves"] = np.array(locs_O)
    fiducials["S_waves"] = np.array(locs_S)
    fiducials["N_waves"] = np.array(locs_N)
    fiducials["D_waves"] = np.array(locs_D)

    return fiducials


def correct_missing_duplicate_peaks(locs_valleys: ArrayLike, locs_peaks: ArrayLike, peaks: ArrayLike) -> tuple:
    """Detects missing or duplicate peaks in a given peak array using PPG onset locations as reference."""
    search_ref = np.array(locs_valleys)
    loc_ = []
    amp_ = []
    locs_peaks = np.array(locs_peaks)
    j = 0
    for i in range(len(search_ref) - 1):
        ind_ = np.asarray(np.where((search_ref[i] < locs_peaks) & (locs_peaks < search_ref[i + 1])))

        if np.size(ind_) == 0:
            amp_.insert(i, np.NaN)
            amp_.insert(i, np.NaN)
            j = j + 1
        elif np.size(ind_) == 1:
            amp_.insert(i, peaks[j])
            loc_.insert(i, locs_peaks[j])
            j += 1
        else:
            peak_mx = np.max(peaks[ind_])
            ind_mx = np.argmax(peaks[ind_])
            amp_.insert(i, peak_mx)
            loc_.insert(i, locs_peaks[ind_[ind_mx][0]])
            j += np.size(ind_)

    return np.array(loc_), amp_


def _generate_search_indices(w_len: int, sig_len: int) -> ArrayLike:
    """Generates search indices for fiducial search."""
    ind = []
    for k1 in range(0, sig_len - w_len + 1, w_len - 2):
        ind_ = np.arange(k1, k1 + w_len, 1)
        ind.append(ind_)

    return ind


def _search_slope_reversals(
    sig: ArrayLike,
    direction: str,
    search_direction: str,
    criterion: str,
    search_indices: ArrayLike = None,
    search_start: ArrayLike = None,
    search_end: ArrayLike = None,
) -> ArrayLike:
    """Searches for a slope reversal point in the given array."""

    if search_indices is None:
        # Generate search indices from search_start and search_end
        search_indices = [np.arange(start, end) for start, end in zip(search_start, search_end)]

    locs = []
    for i in range(len(search_indices)):
        ind_seg = search_indices[i]
        sig_seg = sig[ind_seg]

        loc_rev = _find_slope_reversals(
            sig_seg, direction=direction, search_direction=search_direction, criterion=criterion
        )
        if np.size(loc_rev) != 0:
            loc = ind_seg[loc_rev]
            locs = np.append(locs, loc).astype(int)

    return locs


def _find_slope_reversals(
    sig: ArrayLike, direction: str = "both", search_direction: str = "left_to_right", criterion: str = "all"
) -> ArrayLike:
    """Detects slope reversal points for the given direction and search direction, and returns the required ones according to the selected criterion."""
    # Find the slopes of the signal by taking the difference between adjacent elements
    slopes = np.diff(sig)

    # Find the indices of the slope reversal points based on the specified direction and search direction
    if direction == "positive":
        if search_direction == "left_to_right":
            indices = np.where((slopes[:-1] < 0) & (slopes[1:] >= 0))[0] + 1
        elif search_direction == "right_to_left":
            indices = np.where((slopes[:-1] >= 0) & (slopes[1:] < 0))[0] + 1
        else:
            raise ValueError("Undefined search direction!")

    elif direction == "negative":
        if search_direction == "left_to_right":
            indices = np.where((slopes[:-1] > 0) & (slopes[1:] <= 0))[0] + 1
        elif search_direction == "right_to_left":
            indices = np.where((slopes[:-1] <= 0) & (slopes[1:] > 0))[0] + 1
        else:
            raise ValueError("Undefined search direction!")

    elif direction == "both":
        if search_direction == "left_to_right":
            indices = np.where(np.sign(slopes[:-1]) != np.sign(slopes[1:]))[0] + 1
        elif search_direction == "right_to_left":
            indices = np.where(np.sign(slopes[::-1][:-1]) != np.sign(slopes[::-1][1:]))[0][::-1] + 1
        else:
            raise ValueError("Undefined search direction!")

    else:
        raise ValueError("Undefined direction!")

    # Return the required slope reversal points based on the criterion
    if criterion == "all":
        return indices
    elif criterion == "first":
        return indices[0] if len(indices) > 0 else []
    elif criterion == "max":
        return indices[np.argmax(sig[indices])] if len(indices) > 0 else []
    elif criterion == "min":
        return indices[np.argmin(sig[indices])] if len(indices) > 0 else []
    else:
        raise ValueError("Undefined criterion!")


def _search_zero_crossings(
    sig: ArrayLike,
    direction: str,
    search_direction: str,
    criterion: str,
    search_start: ArrayLike = None,
    search_end: ArrayLike = None,
    search_indices: ArrayLike = None,
) -> ArrayLike:
    """Searches for a zero crossing point in the given array."""
    if search_indices is None:
        # Generate search indices from search_start and search_end
        search_indices = [np.arange(start, end) for start, end in zip(search_start, search_end)]

    zero_cross = []

    for n in range(len(search_start)):
        search_int = search_indices[n]
        vpg_rev = []

        if search_direction == "left_to_right":
            for p in range(len(search_int) - 1):
                if direction == "positive":
                    if (sig[search_int[p]] < 0) and (sig[search_int[p + 1]] >= 0):
                        vpg_rev.append([search_int[p], sig[search_int[p]]])
                        if criterion == "first":
                            break

                elif direction == "negative":
                    if (sig[search_int[p]] > 0) and (sig[search_int[p + 1]] <= 0):
                        vpg_rev.append([search_int[p], sig[search_int[p]]])
                        if criterion == "first":
                            break

                elif direction == "both":
                    if (sig[search_int[p]] == 0) and (sig[search_int[p + 1]] != 0):
                        vpg_rev.append([search_int[p], sig[search_int[p]]])
                        if criterion == "first":
                            break

                else:
                    raise ValueError("Undefined direction.")

        elif search_direction == "right_to_left":
            for p in reversed(range(len(search_int) - 1)):
                if direction == "positive":
                    if (sig[search_int[p]] < 0) and (sig[search_int[p + 1]] >= 0):
                        vpg_rev.append([search_int[p], sig[search_int[p]]])
                        if criterion == "first":
                            break

                elif direction == "negative":
                    if (sig[search_int[p]] > 0) and (sig[search_int[p + 1]] <= 0):
                        vpg_rev.append([search_int[p], sig[search_int[p]]])
                        if criterion == "first":
                            break

                elif direction == "both":
                    if (sig[search_int[p]] == 0) and (sig[search_int[p + 1]] != 0):
                        vpg_rev.append([search_int[p], sig[search_int[p]]])
                        if criterion == "first":
                            break

                else:
                    raise ValueError("Undefined direction.")

        else:
            raise ValueError("Undefined search direction.")

        if criterion == "last":
            if len(vpg_rev) > 0:
                zero_cross.append(vpg_rev[-1])
            else:
                zero_cross.append([len(sig), sig[-1]])
        else:
            zero_cross.extend(vpg_rev)

    return zero_cross
