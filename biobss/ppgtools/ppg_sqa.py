import numpy as np
import math
from numpy.typing import ArrayLike
from typing import Tuple

def detect_clipping(ppg_sig: ArrayLike, threshold: float) -> ArrayLike:
    """Decides if clipping occurs and marks start and end of the clipped part.
    
    Args:
        ppg_sig (Array): Signal segment to be processed
        threshold (float, optional): The threshold for clipping. 

    Returns:
        Array: The sample numbers of the start and end of the clipped parts
    """

    #Copied from HeartPy
    clip_binary = np.where(ppg_sig > threshold)
    clipping_edges = np.where(np.diff(clip_binary) > 1)[1]
    clipped_segments = []

    for i in range(0, len(clipping_edges)):
        if i == 0: #if first clipping segment
            clipped_segments.append((clip_binary[0][0], 
                                      clip_binary[0][clipping_edges[0]]))
        elif i == len(clipping_edges) - 1:
            #append last entry
            clipped_segments.append((clip_binary[0][clipping_edges[i]+1],
                                      clip_binary[0][-1]))    
        else:
            clipped_segments.append((clip_binary[0][clipping_edges[i-1] + 1],
                                      clip_binary[0][clipping_edges[i]]))

    return clipped_segments


def detect_flatline(ppg_sig: ArrayLike, threshold: float, duration: float) -> ArrayLike:
    """Detects flatlines in a signal. 

    Args:
        ppg_sig (Array): Signal segment to be processed
        threshold (float): The tolerance in amplitude difference to accept a signal part as stationary.
        duration (float): The duration (number of samples) of stationary parts required to be marked as flatline.

    Returns:
        Array: The sample numbers of the start and end of the flatline segments
    """

    sig_dif=np.diff(ppg_sig)

    flat_binary = np.where(abs(sig_dif) < threshold)
    flat_edges = np.where(np.diff(flat_binary) > 1)[1]

    #Copied from HeartPy
    flat_segments = []
    
    for i in range(0, len(flat_edges)):
        if i == 0: #if first flat segment
            flat_segments.append((flat_binary[0][0], 
                                      flat_binary[0][flat_edges[0]]))
        elif i == len(flat_edges) - 1:
            #append last entry
            flat_segments.append((flat_binary[0][flat_edges[i]+1],
                                      flat_binary[0][-1]))    
        else:
            flat_segments.append((flat_binary[0][flat_edges[i-1] + 1],
                                      flat_binary[0][flat_edges[i]]))

    flatline_segments=[]

    for j in range(len(flat_segments)):
        flat_dur=flat_segments[j][1]-flat_segments[j][0]
        if flat_dur >= duration:
            flatline_segments.append(flat_segments[j])

    return flatline_segments


def check_phys(peaks_locs: ArrayLike, sampling_rate:float) -> dict:
    """Checks for physiological viability.

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

    if (HR_mean < 40 or HR_mean > 180):
        info['Rule 1']=False
    else:
        info['Rule 1']=True
        
    #Rule 2: Maximum P-P interval: 1.5 seconds. Allowing for a single missing beat, it is 3 seconds
    ind= np.where(intervals > 3)

    if np.size(ind)>0:
        info['Rule 2']=False
    else:
        info['Rule 2']=True

    #Rule 3: Maximum P-P interval / minimum P-P interval ratio: 10 of the signal length for a short signal. 
             #For 10 seconds signal, it is 1.1; allowing for a single missing beat, it is 2.2  

    if (intervals.max()/intervals.min())> 2.2:
        info['Rule 3']=False
    else:
        info['Rule 3']=True

    return info


def check_morph(peaks_locs: ArrayLike ,peaks_amps: ArrayLike, troughs_locs: ArrayLike, troughs_amps: ArrayLike,sampling_rate:float) -> dict:
    """Checks for ranges of morphological features

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

    #Rule 1: Systolic phase duration(rise time): 0.08 to 0.49 s
    SP= (peaks_locs-troughs_locs[:-1])/sampling_rate
    ind1= np.where(SP<0.08)
    ind2= np.where(SP>0.49)

    if np.size(ind1)>0 or np.size(ind2)>0:
        info['Rule 1']=False
    else:
        info['Rule 1']=True

    #Rule 2: Ratio of systolic phase duration to diastolic phase duration: max 1.1
    DP= (troughs_locs[1:]-peaks_locs)/sampling_rate #Diastolic phase duration
    SP_DP = SP/DP
    ind=np.where( SP_DP > 1.1)

    if np.size(ind)>0:
        info['Rule 2']=False
    else:
        info['Rule 2']=True

    #Rule 3: Pulse wave duration: 0.27 to 2.4 s
    PWD = np.diff(troughs_locs)/sampling_rate
    ind1=np.where(PWD < 0.27)
    ind2=np.where(PWD > 2.4)

    if np.size(ind1)>0 or np.size(ind2)>0:
        info['Rule 3']=False
    else:
        info['Rule 3']=True

    #Rule 4: Variation in PWD and SP: 33-300%
    var_SP= (np.max(peaks_amps)-np.min(peaks_amps))/np.min(peaks_amps)*100
    var_PWD= (np.max(PWD)-np.min(PWD))/np.min(PWD)*100

    if ( var_SP > 300 ) or ( var_PWD > 300 ):
        info['Rule 4']=False
    else:
        info['Rule 4']=True

    #Rule 5: Variation in PWA: 25-400% (Pulse wave amplitude: a threshold which was set heuristically)
    PWA = peaks_amps-troughs_amps[:-1]
    var_PWA= (np.max(PWA)-np.min(PWA))/np.min(PWA)*100

    if ( var_PWA > 400  ):
        info['Rule 5']=False
    else:
        info['Rule 5']=True

    return info


def template_matching(ppg_sig: ArrayLike, peaks_locs:ArrayLike, corr_th:float=0.9) -> Tuple[float,bool]:
    """Applies template matching method for signal quality assessment

    Args:
        ppg_sig (Array): _description_
        peaks_locs (Array): _description_
        corr_th (float, optional): Threshold for the correlation coefficient above which the signal is considered to be valid. Defaults to 0.9.

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