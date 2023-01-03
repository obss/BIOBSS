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
    #_find_missing_duplicate_peaks(locs_valleys=troughs_locs, locs_peaks=peaks_locs, peaks=peaks_amp)
    
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


def ppg_waves(sig:ArrayLike, locs_onsets:ArrayLike, sampling_rate:float, th_w:float=0.5, th_y:float=0.45, th_a:float=0.45):

    vpg_sig = np.gradient(sig) / (1/sampling_rate)
    apg_sig = np.gradient(vpg_sig) / (1/sampling_rate)

    fiducials = {}

    vpg_fiducials = vpg_delineate(vpg_sig=vpg_sig, sampling_rate=sampling_rate, th_w=th_w, th_y=th_y)
    apg_fiducials= apg_delineate(apg_sig=apg_sig, vpg_sig=vpg_sig, vpg_fiducials=vpg_fiducials, sampling_rate=sampling_rate,th_a=th_a, th_w=th_w)
    ppg_fiducials = ppg_delineate(ppg_sig=sig, vpg_sig=vpg_sig, vpg_fiducials=vpg_fiducials, apg_sig=apg_sig, apg_fiducials=apg_fiducials, sampling_rate=sampling_rate, locs_onsets=locs_onsets)
   
    fiducials.update(ppg_fiducials)
    fiducials.update(vpg_fiducials)
    fiducials.update(apg_fiducials)

    return fiducials

def vpg_delineate(vpg_sig:ArrayLike, sampling_rate:float, th_w:float=0.5, th_y:float=0.45) -> dict:

    fiducials = {}

    ###########################################
    #Detect w-waves (maximum peak of VPG signal)
    ###########################################
    
    max_amp = np.max(vpg_sig)
    thr_w = th_w * max_amp 

    cropped_vpg = vpg_sig.copy()
    cropped_vpg[vpg_sig<thr_w] = 0

    w_len = round(0.25*sampling_rate)

    #Generate the search indices
    ind = _generate_search_indices(w_len=w_len, sig_len=len(cropped_vpg))

    #Search for a slope reversal point in each window (w_len)
    locs_w = _search_slope_reversals(sig=cropped_vpg, search_indices=ind, direction='negative', search_direction='left_to_right')
    peaks_w = vpg_sig[locs_w]

    ###########################################
    #Detect y-waves (minimum trough of VPG signal)
    ###########################################

    min_amp = min(vpg_sig)
    thr_y = th_y * min_amp

    #Define a search interval
    search_st= locs_w
    locs = np.append(locs_w,len(vpg_sig))
    search_int=np.round(np.diff(locs)/2).astype(int)
    search_end=search_st+search_int
    cropped_vpg = np.zeros(len(vpg_sig))

    for k in range(len(search_int)):
        cropped_vpg[search_st[k]:search_end[k]] = vpg_sig[search_st[k]:search_end[k]]

    cropped_vpg[vpg_sig>thr_y] = 0

    #Generate the search indices
    ind = _generate_search_indices(w_len=w_len, sig_len=len(cropped_vpg))

    #Search for a slope reversal in each window (w_len)
    locs_y = _search_slope_reversals(sig=cropped_vpg, search_indices=ind, direction='positive', search_direction='left_to_right')
    peaks_y = vpg_sig[locs_y]

    #Select the minimum one if there are more than one slope reversal in each window
    loc_y=[]
    peak_y=[]
    search_end=np.append(locs_w, len(vpg_sig))
    for c in range(1, len(search_end)):
        ind_y=np.where((locs_y < search_end[c]) & (locs_y > search_end[c-1]))[0]
        if len(ind_y) != 0:
            peak = np.min(peaks_y[ind_y])
            loc = np.argmin(peaks_y[ind_y])

            loc_y=np.append(loc_y,locs_y[ind_y[loc]])
            peak_y=np.append(peak_y,peak)

    peaks_y=peak_y.copy()
    locs_y=loc_y.astype(int).copy()

    ###########################################
    #Detect z-waves (local extreme)
    ###########################################

    #search for the next zero crossing point
    search_st = locs_y + 1
    search_end = np.append(search_st[1:],len(vpg_sig))
    
    zero_cross = _search_zero_crossings(sig=vpg_sig, search_st=search_st, search_end=search_end, direction='positive', search_direction='left_to_right')

    #if more than one zero-crossing point was found in each cycle
    if len(zero_cross) != len(locs_y):
        if locs_y[0] < zero_cross[0]:
            locs_y=locs_y[:len(zero_cross)]
            peaks_y=peaks_y[:len(zero_cross)]
        else:
            zero_cross=zero_cross[1:]   

    #modify the signal (first 70% percent of the samples)
    zero_cross_loc=np.array([x[0] for x in zero_cross])
    nofsamp = np.round(0.7 * (zero_cross_loc - locs_y)).astype(int)

    a1 = locs_y
    b1 = peaks_y
    a2 = locs_y + nofsamp
    b2 = vpg_sig[a2]

    m = (b2 - b1) / (a2 - a1)

    mod_samp = vpg_sig.copy()

    max_ind = []
    for j in range(len(m)):
        i = 0
        for k in range(a1[j], a2[j]+1):       
            mod_samp[k] = vpg_sig[k] - m[j] * i - b1[j]
            i += 1

        mod_max = np.max(mod_samp[a1[j]:a2[j]+1])  # Find the maximum
        ind = np.argmax(mod_samp[a1[j]:a2[j]+1])  # Find the maximum
        ind += a1[j]
        max_ind.append(ind)    

    locs_z=np.array(max_ind)
    peaks_z=vpg_sig[max_ind]


    fiducials['w_waves'] = locs_w
    fiducials['y_waves'] = locs_y
    fiducials['z_waves'] = locs_z

    return fiducials

def apg_delineate(apg_sig:ArrayLike, vpg_sig:ArrayLike, vpg_fiducials:dict, sampling_rate:float, th_a:float=0.45, th_w:float=0.5) -> dict:

    locs_w=vpg_fiducials['w_waves']
    locs_y=vpg_fiducials['y_waves']
    locs_z=vpg_fiducials['z_waves']

    fiducials = {}    

    ###########################################
    #Detect a-waves (maximum peak of APG signal)
    ###########################################

    max_amp = np.max(apg_sig)
    thr_a = th_a * max_amp

    cropped_apg = apg_sig.copy()
    cropped_apg[apg_sig<thr_a] = 0

    w_len = round(0.25 * sampling_rate)

    thr_w = th_w * np.max(vpg_sig)
    cropped_vpg = vpg_sig.copy()
    cropped_vpg[vpg_sig<thr_w] = 0

    #Generate the search indices
    ind = _generate_search_indices(w_len, sig_len=len(cropped_vpg))

    #Search for a slope reversal point in each segment (w_len)

    #it should be 'negative' and 'left_to_right' but the equality checks match better for this combination
    #check the paper and consider refactoring _find_slope_reversals
    locs_a = _search_slope_reversals(sig=cropped_apg, search_indices=ind, direction='positive', search_direction='right_to_left')
    peaks_a = apg_sig[locs_a]

    ###########################################
    #Detect b-waves (minimum trough of APG signal)
    ###########################################

    # Search for the next zero-crossing point
    search_st = locs_a + 1
    search_end = np.append(search_st[1:], len(apg_sig))

    zero_crossings=_search_zero_crossings(sig=apg_sig, search_st=search_st, search_end=search_end, direction='negative', search_direction='left_to_right')
    zero_cross=np.array([x[0] for x in zero_crossings])

    #search for the first slope reversal point

    peaks_b = []
    locs_b = []

    search_int=np.append(zero_cross, len(apg_sig))

    for m in range(len(search_int) - 1):
        slope = np.diff(apg_sig[search_int[m]:search_int[m + 1]])
        p = 1
        flag = 0

        while flag == 0 and p < len(slope):
            if slope[p] < 0 and slope[p + 1] >= 0:
                loc_b = zero_cross[m] + p
                peak_b = apg_sig[loc_b]
                peaks_b.append(peak_b)
                locs_b.append(loc_b)
                flag = 1
            p += 1

    ###########################################
    #Detect e-waves 
    ###########################################

    #Generate search indices
    ind=[]
    for k in range(len(locs_z)):
        ind.append(np.arange(locs_y[k] - 5,locs_z[k] + 6))

    # Search for a slope reversal point (from z to y, on APG)
    locs_e = []
    peaks_e = []

    for k in range(len(ind)):
        seg = apg_sig[ind[k]]
        slope = np.diff(seg)

        p = len(slope)
        flag = 0

        while flag == 0 and p > 1:
            if slope[p - 1] < 0 and slope[p - 2] >= 0:
                loc_e = locs_y[k] - 5 + p - 1
                peak_e = apg_sig[loc_e]
                peaks_e.append(peak_e)
                locs_e.append(loc_e)
                flag = 1
            p -= 1

    ###########################################
    #Detect c and d-waves 
    ###########################################

    # 1. Search for slope reversal points (from e to b)
    # First: d, second: c

    locs_c = []
    peaks_c = []
    locs_d = []
    peaks_d = []

    for h in range(len(locs_e)):
        seg = apg_sig[locs_b[h] + 1:locs_e[h] - 1]
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
            #If no slope reversal point is found on
            #the SDPPG data, then starting from the location of b peak, all the
            #samples up to the location of e peak are modified (Equation 4).

            a1=locs_b[h]
            b1=peaks_b[h]
            a2=locs_e[h]
            b2=peaks_e[h]

            m=(b2-b1)/(a2-a1)
            mod_samp=apg_sig[a1:a2]
            
            mod_samp = np.empty(len(range(a1, a2)))

            for i, k in enumerate(range(a1, a2)):
                mod_samp[i] = apg_sig[k] - m * i

            
            mod_max=np.max(mod_samp)
            ind=np.argmax(mod_samp)
            max_ind=ind+a1        

            #Now, among these modified APG signals, the maximum peak location
            #is determined at first. Then takin reference to the location of
            #this maximum peak, a right traversal is carried out on those
            #modified APG samples to find out the first slope change point as
            #the location of c peak and the next slope change point as the
            #location of d peak.T

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

    fiducials['a_waves'] = np.asarray(locs_a)
    fiducials['b_waves'] = np.asarray(locs_b)
    fiducials['e_waves'] = np.asarray(locs_e)
    fiducials['c_waves'] = np.asarray(locs_c)
    fiducials['d_waves'] = np.asarray(locs_d)

    return fiducials

def ppg_delineate(ppg_sig:ArrayLike, vpg_sig:ArrayLike, vpg_fiducials:dict, apg_sig:ArrayLike, apg_fiducials:dict, sampling_rate:float, locs_onsets:ArrayLike=None) -> dict:

    locs_w = vpg_fiducials['w_waves']
    locs_y = vpg_fiducials['y_waves']
    locs_z = vpg_fiducials['z_waves']

    locs_e = apg_fiducials['e_waves']

    fiducials = {}

    ###########################################
    #Detect O-waves (pulse onsets) 
    ###########################################

    if locs_onsets is None:
        info = peak_detection(ppg_sig, sampling_rate,'peakdet', delta=0.01)
        locs_O=info['Trough_locs']
        locs_peaks=info['Peak_locs']

        info2 = peak_control(ppg_sig, peaks_locs=locs_peaks, troughs_locs=locs_O, type='peak')
        locs_O=info2['Trough_locs']
        locs_peaks=info2['Peak_locs']

    else:
        locs_O = locs_onsets


    ###########################################
    #Detect S-waves (systolic peaks) 
    ###########################################    

    # Search for the first zero crossing poing (from w to the right)

    search_st = locs_w + 1
    zero_cross = []

    for i in range(len(search_st)):
        p = search_st[i]
        flag = 0

        while flag == 0:
            if vpg_sig[p] > 0 and vpg_sig[p + 1] <= 0:
                zero_cross.append(p)
                flag = 1
            p += 1

    zero_cross = np.array(zero_cross)

    #Select twenty samples of the PPG signal around each of zero crossing point.
    
    search_int=np.array([[x-10,x+10] for x in zero_cross])

    #Then, find the slope reversal point among these marked samples

    locs_S = []
    peaks_S = []

    for r in range(len(search_int)):
        seg = ppg_sig[search_int[r][0]:search_int[r][1]]
        slope = np.diff(seg)

        loc = []
        peak = []

        for j in range(len(slope) - 1):
            if (slope[j] >= 0) and (slope[j + 1] <= 0):
                loc.append(search_int[r][0] + j+1)
                peak.append(ppg_sig[search_int[r][0] + j+1])

        mn = min(abs(np.array(loc) - zero_cross[r]))
        in_ = np.argmin(abs(np.array(loc) - zero_cross[r]))
        loc_S = loc[in_]
        peak_S = ppg_sig[loc_S]

        locs_S.append(loc_S)
        peaks_S.append(peak_S)

    ###########################################
    #Detect N-waves (dicrotic notches) 
    ########################################### 

    locs_N = locs_e
    peaks_N = ppg_sig[locs_N]   

    ###########################################
    #Detect D-waves (diastolic peaks) 
    ###########################################

    #Search for a slope reversal point (from e to the right)
    locs_D = []
    peaks_D = []

    search_st = locs_e+1
    search_end = np.append(locs_e[1:], len(apg_sig))

    for i in range(len(search_st)):

        seg = apg_sig[search_st[i]:search_end[i]]
        slope = np.diff(seg)
        p = 1
        flag = 0

        while flag == 0:
            if slope[p] < 0 and slope[p+1] >= 0:
                loc_D = search_st[i] + p - 2
                peak_D = ppg_sig[loc_D]
                peaks_D.append(peak_D)
                locs_D.append(loc_D)

                flag = 1
            p += 1    

    fiducials['O_waves'] = np.array(locs_O)
    fiducials['S_waves'] = np.array(locs_S)
    fiducials['N_waves'] = np.array(locs_N)
    fiducials['D_waves'] = np.array(locs_D)

    return fiducials

def find_missing_duplicate_peaks(locs_valleys: ArrayLike, locs_peaks:ArrayLike, peaks:ArrayLike) -> tuple:
    search_peaks = locs_valleys
    loc_peaks = []
    peak_peaks = []
    j = 0
    for i in range(len(search_peaks)-1):
        ind_peaks = [j for j in range(len(locs_peaks)) if search_peaks[i] < locs_peaks[j] < search_peaks[i+1]]
        if not ind_peaks:
            peak_peaks.append(float('nan'))
            loc_peaks.append(float('nan'))
        elif len(ind_peaks) == 1:
            peak_peaks.append(peaks[j])
            loc_peaks.append(locs_peaks[j])
            j += 1
        else:
            peak_mx = max(peaks[j] for j in ind_peaks)
            ind_mx = ind_peaks[peaks[j] == peak_mx]
            peak_peaks.append(peak_mx)
            loc_peaks.append(locs_peaks[ind_mx])
            j += 1
    return np.array(loc_peaks), peak_peaks
    
def _generate_search_indices(w_len: int, sig_len: int) -> ArrayLike:

    ind=[]
    for k1 in range(0, sig_len-w_len+1, w_len-2):
        ind_=np.arange(k1,k1+w_len,1)
        ind.append(ind_)

    return ind


def _search_slope_reversals(sig:ArrayLike, search_indices:ArrayLike, direction:str, search_direction:str) -> ArrayLike:
    #Search for a slope reversal point in each window (w_len)

    locs = []
    
    for i in range(len(search_indices)-1):
        ind_seg=search_indices[i]
        sig_seg=sig[ind_seg]

        loc_rev = _find_slope_reversals(sig_seg, direction=direction, search_direction=search_direction)
        if len(loc_rev) != 0:
            loc = ind_seg[loc_rev]
            locs=np.append(locs, loc).astype(int)

    return locs

def _find_slope_reversals(sig:ArrayLike, direction:str='both', search_direction: str='left_to_right') -> ArrayLike:

    # Find the slopes of the signal by taking the difference between adjacent elements
    slopes = np.diff(sig)
    
    # Initialize an empty list to store the indices of the slope reversal points
    indices = []
    
    # Find the indices of the slope reversal points based on the specified direction and search direction
    if direction == 'positive':
        if search_direction == 'left_to_right':
            indices = np.where((slopes[:-1] < 0) & (slopes[1:] >= 0))[0] + 1
        elif search_direction == 'right_to_left':
            indices = np.where((slopes[:-1] >= 0) & (slopes[1:] < 0))[0] + 1
        else:
            raise ValueError("Undefined search direction!")

    elif direction == 'negative':
        if search_direction == 'left_to_right':
            indices = np.where((slopes[:-1] > 0) & (slopes[1:] <= 0))[0] + 1
        elif search_direction == 'right_to_left':
            indices = np.where((slopes[:-1] <= 0) & (slopes[1:] > 0))[0] + 1
        else:
            raise ValueError("Undefined search direction!")

    elif direction == 'both':
        if search_direction == 'left_to_right':
            indices = np.where(np.sign(slopes[:-1]) != np.sign(slopes[1:]))[0] + 1
        elif search_direction == 'right_to_left':
            indices = np.where(np.sign(slopes[::-1][:-1]) != np.sign(slopes[::-1][1:]))[0][::-1] + 1
        else:
            raise ValueError("Undefined search direction!")

    else:
        raise ValueError("Undefined direction!")

    return indices

def _search_zero_crossings(sig:ArrayLike, search_st:ArrayLike, search_end:ArrayLike, direction:str, search_direction:str) -> ArrayLike:

    zero_cross = []

    for n in range(len(search_st)):
        search_int = np.arange(search_st[n],search_end[n])
        vpg_rev = []
        flag = 0

        if search_direction == 'left_to_right':
            for p in range(len(search_int)-1):
                if direction == 'positive':
                    if (sig[search_int[p]] < 0) and (sig[search_int[p+1]] >= 0) and (flag == 0):
                        vpg_rev.append([search_int[1] + p - 1, sig[search_int[p]]])
                        flag = 1
                elif direction == 'negative':
                    if (sig[search_int[p]] > 0) and (sig[search_int[p+1]] <= 0) and (flag == 0):
                        vpg_rev.append([search_int[1] + p - 1, sig[search_int[p]]])
                        flag = 1
                else:
                    raise ValueError("Undefined direction.")

        elif search_direction == 'right_to_left':
            for p in reversed(range(len(search_int)-1)):
                if direction == 'positive':
                    if (sig[search_int[p]] < 0) and (sig[search_int[p+1]] >= 0) and (flag == 0):
                        vpg_rev.append([search_int[1] + p - 1, sig[search_int[p]]])
                        flag = 1
                elif direction == 'negative':
                    if (sig[search_int[p]] > 0) and (sig[search_int[p+1]] <= 0) and (flag == 0):
                        vpg_rev.append([search_int[1] + p - 1, sig[search_int[p]]])
                        flag = 1                                                       
                else:
                    raise ValueError("Undefined direction.")
        
        else: 
            raise ValueError("Undefined search direction.")

        if (n == len(search_st)) and (len(vpg_rev) == 0):
            vpg_rev.append([len(sig), sig[-1]])

        zero_cross.append(vpg_rev[-1])

    return zero_cross
