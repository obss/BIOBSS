import numpy as np
import math
from numpy.typing import ArrayLike
from typing import Tuple

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

def detect_flatline_clipping(ppg_sig: ArrayLike, threshold: float, clipping:bool=True, flatline:bool=False, **kwargs) -> dict:
    """Detects flatlines and clipped parts of the signal.

    Args:
        ppg_sig (ArrayLike): PPG signal to be analyzed.
        threshold (float): Threshold value for clipping/flatline.
        clipping (bool, optional): True for clipping detection. Defaults to True.
        flatline (bool, optional): True for flatline detection. Defaults to False.
        **kwargs (dict): Keyword arguments

    Keyword Args:
        duration (float): Mimimum duration of flat segments for flatline detection.

    Raises:
        ValueError: If keyword argument 'duration' is not given.

    Returns:
        dict: Dictionary of boundaries of clipped and/or flatline segments.
    """
    
    info={}

    if clipping:
        clip_binary = np.where(ppg_sig > threshold)
        clipped_segments=_detect_flat_segments(clip_binary)
        info['Clipped segments']=clipped_segments

    if flatline:
        if 'duration' in kwargs:
            sig_dif=np.diff(ppg_sig)
            flat_binary = np.where(abs(sig_dif) < threshold)
            flat_segments=_detect_flat_segments(flat_binary)

            flatline_segments=[]
            for j in range(len(flat_segments)):
                flat_dur=flat_segments[j][1]-flat_segments[j][0]
                if flat_dur >= kwargs['duration']:
                    flatline_segments.append(flat_segments[j])

            info['Flatline segments']=flatline_segments
        
        else:
            raise ValueError('Flatline detection requires a keyword argument: duration')
        
    return info


def _detect_flat_segments(binary_array):

    #Copied from HeartPy
    edges = np.where(np.diff(binary_array) > 1)[1]
    segments = []

    for i in range(0, len(edges)):
        if i == 0: #if first flat segment
            segments.append((binary_array[0][0], 
                                      binary_array[0][edges[0]]))
        elif i == len(edges) - 1:
            #append last entry
            segments.append((binary_array[0][edges[i]+1],
                                      binary_array[0][-1]))    
        else:
            segments.append((binary_array[0][edges[i-1] + 1],
                                      binary_array[0][edges[i]]))    

    return segments


def check_phys(peaks_locs: ArrayLike, sampling_rate:float) -> dict:
    """Checks for physiological viability.

    Rule 1: Average HR should be between 40-180 bpm (up to 300 bpm in the case of exercise)
    Rule 2: Maximum P-P interval: 1.5 seconds. Allowing for a single missing beat, it is 3 seconds
    Rule 3: Maximum P-P interval / minimum P-P interval ratio: 10 of the signal length for a short signal. 
            For 10 seconds signal, it is 1.1; allowing for a single missing beat, it is 2.2 

    Args:
        peaks_locs (Array): Peak locations
        sampling_rate (float): Sampling rate of the PPG signal

    Returns:
        dict: Dictionary of decisions.
    """

    info={}

    #Rule 1: Average HR should be between 40-180 bpm (up to 300 bpm in the case of exercise)
    intervals = np.diff(peaks_locs)/sampling_rate
    HR_mean = 60 / np.mean(intervals)

    if (HR_mean < HR_MIN or HR_mean > HR_MAX):
        info['Rule 1']=False
    else:
        info['Rule 1']=True
        
    #Rule 2: Maximum P-P interval: 1.5 seconds. Allowing for a single missing beat, it is 3 seconds
    if np.size(np.where(intervals > PP_MAX))>0:
        info['Rule 2']=False
    else:
        info['Rule 2']=True

    #Rule 3: Maximum P-P interval / minimum P-P interval ratio: 10 of the signal length for a short signal. 
             #For 10 seconds signal, it is 1.1; allowing for a single missing beat, it is 2.2  
    if (intervals.max()/intervals.min())> MAX_PP_RATIO:
        info['Rule 3']=False
    else:
        info['Rule 3']=True

    return info


def check_morph(peaks_locs: ArrayLike ,peaks_amps: ArrayLike, troughs_locs: ArrayLike, troughs_amps: ArrayLike,sampling_rate:float) -> dict:
    """Checks for ranges of morphological features.

    Rule 1: Systolic phase duration(rise time): 0.08 to 0.49 s
    Rule 2: Ratio of systolic phase duration to diastolic phase duration: max 1.1
    Rule 3: Pulse wave duration: 0.27 to 2.4 s
    Rule 4: Variation in PWD and SP: 33-300%
    Rule 5: Variation in PWA: 25-400% (Pulse wave amplitude: a threshold which was set heuristically)

    Args:
        peaks_locs (Array): Peak locations
        peaks_amps (Array): Peak amplitudes
        troughs_locs (Array): Trough locations
        troughs_amps (Array): Trough amplitudes
        sampling_rate (float): Sampling rate of the PPG signal.

    Returns:
        dict: Dictionary of decisions
    """
    
    info={}

    #Rule 1
    SP= (peaks_locs-troughs_locs[:-1])/sampling_rate

    if np.size(np.where(SP<MIN_SPD))>0 or np.size(np.where(SP>MAX_SPD))>0:
        info['Rule 1']=False
    else:
        info['Rule 1']=True

    #Rule 2:
    DP= (troughs_locs[1:]-peaks_locs)/sampling_rate #Diastolic phase duration
    SP_DP = SP/DP

    if np.size(np.where(SP_DP > SP_DP_RATIO))>0:
        info['Rule 2']=False
    else:
        info['Rule 2']=True

    #Rule 3:
    PWD = np.diff(troughs_locs)/sampling_rate

    if np.size(np.where(PWD < MIN_PWD))>0 or np.size(ind2=np.where(PWD > MAX_PWD))>0:
        info['Rule 3']=False
    else:
        info['Rule 3']=True

    #Rule 4:
    var_SP= (np.max(peaks_amps)-np.min(peaks_amps))/np.min(peaks_amps)*100
    var_PWD= (np.max(PWD)-np.min(PWD))/np.min(PWD)*100

    if ( var_SP > MAX_VAR_DUR ) or ( var_PWD > MAX_VAR_DUR ):
        info['Rule 4']=False
    else:
        info['Rule 4']=True

    #Rule 5: 
    PWA = peaks_amps-troughs_amps[:-1]
    var_PWA= (np.max(PWA)-np.min(PWA))/np.min(PWA)*100

    if ( var_PWA > MAX_VAR_AMP ):
        info['Rule 5']=False
    else:
        info['Rule 5']=True

    return info


def template_matching(ppg_sig: ArrayLike, peaks_locs:ArrayLike, corr_th:float=CORR_TH) -> Tuple[float,bool]:
    """Applies template matching method for signal quality assessment

    Args:
        ppg_sig (ArrayLike): Signal to be analyzed.
        peaks_locs (ArrayLike): Peak locations
        corr_th (float, optional): Threshold for the correlation coefficient above which the signal is considered to be valid. Defaults to CORR_TH.

    Returns:
        Tuple[float,bool]: Correlation coefficient and the decision
    """
 
    wl=np.median(np.diff(peaks_locs))
    waves=np.empty((0,2*math.floor(wl/2)+1))
    nofwaves=np.size(peaks_locs)

    for i in range((nofwaves)):
        wave_st=peaks_locs[i]-math.floor(wl/2)
        wave_end=peaks_locs[i]+math.floor(wl/2)
        wave=[]
    
        if (wave_st < 0):
            wave = ppg_sig[:wave_end]
            for _ in range(-wave_st+1):
                wave=np.insert(wave,0,wave[0])

        elif (wave_end > len(ppg_sig)-1):
            wave = ppg_sig[wave_st-1:]
            for _ in range(wave_end-len(ppg_sig)):
                wave=np.append(wave,wave[-1])      
            
        else:
            wave = ppg_sig[wave_st:wave_end+1]
    
        waves=np.vstack([waves,wave])
        
    sig_temp=np.mean(waves,axis=0)
    
    ps=np.array([])
    for j in range(np.size(peaks_locs)):
        p=np.corrcoef(waves[j],sig_temp,rowvar=True)
        ps=np.append(ps,p[0][1])

    if np.size(np.where(ps<corr_th))>0:
        result=False

    else:
        result=True            

    return ps,result