import numpy as np
from numpy.typing import ArrayLike

#Time domain features
FEATURES_TIME_CYCLE = {
'a_S': lambda _0,peaks,_1,_2,_3: np.mean(peaks),
't_S': lambda _0,_1,locs_peaks,locs_onsets,fs: np.mean((locs_peaks-locs_onsets[:-1])/fs),
't_C': lambda _0,_1,_2,locs_onsets,fs: np.mean(np.diff(locs_onsets)/fs),
'DW': lambda _0,_1,locs_peaks,locs_onsets,fs: np.mean((locs_onsets[1:]-locs_peaks)/fs),
'SW_10': lambda sig,peaks,locs_peaks,locs_onsets,fs: calculate_SW(sig,peaks,locs_peaks,locs_onsets,fs,0.1),
'SW_25': lambda sig,peaks,locs_peaks,locs_onsets,fs: calculate_SW(sig,peaks,locs_peaks,locs_onsets,fs,0.25),
'SW_33': lambda sig,peaks,locs_peaks,locs_onsets,fs: calculate_SW(sig,peaks,locs_peaks,locs_onsets,fs,0.33),
'SW_50': lambda sig,peaks,locs_peaks,locs_onsets,fs: calculate_SW(sig,peaks,locs_peaks,locs_onsets,fs,0.5),
'SW_66': lambda sig,peaks,locs_peaks,locs_onsets,fs: calculate_SW(sig,peaks,locs_peaks,locs_onsets,fs,0.66),
'SW_75': lambda sig,peaks,locs_peaks,locs_onsets,fs: calculate_SW(sig,peaks,locs_peaks,locs_onsets,fs,0.75),
'DW_10': lambda sig,peaks,locs_peaks,locs_onsets,fs: calculate_DW(sig,peaks,locs_peaks,locs_onsets,fs,0.1),
'DW_25': lambda sig,peaks,locs_peaks,locs_onsets,fs: calculate_DW(sig,peaks,locs_peaks,locs_onsets,fs,0.25),
'DW_33': lambda sig,peaks,locs_peaks,locs_onsets,fs: calculate_DW(sig,peaks,locs_peaks,locs_onsets,fs,0.33),
'DW_50': lambda sig,peaks,locs_peaks,locs_onsets,fs: calculate_DW(sig,peaks,locs_peaks,locs_onsets,fs,0.5),
'DW_66': lambda sig,peaks,locs_peaks,locs_onsets,fs: calculate_DW(sig,peaks,locs_peaks,locs_onsets,fs,0.66),
'DW_75': lambda sig,peaks,locs_peaks,locs_onsets,fs: calculate_DW(sig,peaks,locs_peaks,locs_onsets,fs,0.75),
'DW_SW_10': lambda sig,peaks,locs_peaks,locs_onsets,fs: calculate_DW(sig,peaks,locs_peaks,locs_onsets,fs,0.1)/calculate_SW(sig,peaks,locs_peaks,locs_onsets,fs,0.1),
'DW_SW_25': lambda sig,peaks,locs_peaks,locs_onsets,fs: calculate_DW(sig,peaks,locs_peaks,locs_onsets,fs,0.25)/calculate_SW(sig,peaks,locs_peaks,locs_onsets,fs,0.25),
'DW_SW_33': lambda sig,peaks,locs_peaks,locs_onsets,fs: calculate_DW(sig,peaks,locs_peaks,locs_onsets,fs,0.1)/calculate_SW(sig,peaks,locs_peaks,locs_onsets,fs,0.33),
'DW_SW_50': lambda sig,peaks,locs_peaks,locs_onsets,fs: calculate_DW(sig,peaks,locs_peaks,locs_onsets,fs,0.1)/calculate_SW(sig,peaks,locs_peaks,locs_onsets,fs,0.50),
'DW_SW_66': lambda sig,peaks,locs_peaks,locs_onsets,fs: calculate_DW(sig,peaks,locs_peaks,locs_onsets,fs,0.1)/calculate_SW(sig,peaks,locs_peaks,locs_onsets,fs,0.66),
'DW_SW_75': lambda sig,peaks,locs_peaks,locs_onsets,fs: calculate_DW(sig,peaks,locs_peaks,locs_onsets,fs,0.1)/calculate_SW(sig,peaks,locs_peaks,locs_onsets,fs,0.75),
'PR_mean' : lambda _0,_1,locs_peaks,_2,fs: 60/np.mean(np.diff(locs_peaks)/fs),
}

FEATURES_TIME_SEGMENT = {
'zcr': lambda sig,_0: calculate_zcr(sig),
'snr': lambda sig,_0: calculate_snr(sig),
}

def get_time_features(sig: ArrayLike,fs: float,type: str, prefix: str='signal', **kwargs) -> dict:
    """Calculates time-domain features.

    Cycle-based features:
    a_S: Mean amplitude of the systolic peaks 
    t_S: Mean systolic peak duration
    t_C: Mean cycle duration
    DW: Mean diastolic peak duration
    SW_10: The systolic peak duration at 10% amplitude of systolic amplitude
    SW_25: The systolic peak duration at 25% amplitude of systolic amplitude
    SW_33: The systolic peak duration at 33% amplitude of systolic amplitude
    SW_50: The systolic peak duration at 50% amplitude of systolic amplitude
    SW_66: The systolic peak duration at 66% amplitude of systolic amplitude
    SW_75: The systolic peak duration at 75% amplitude of systolic amplitude
    DW_10: The diastolic peak duration at 10% amplitude of systolic amplitude
    DW_25: The diastolic peak duration at 25% amplitude of systolic amplitude
    DW_33: The diastolic peak duration at 33% amplitude of systolic amplitude
    DW_50: The diastolic peak duration at 50% amplitude of systolic amplitude
    DW_66: The diastolic peak duration at 66% amplitude of systolic amplitude
    DW_75: The diastolic peak duration at 75% amplitude of systolic amplitude
    DW_SW_10: The ratio of diastolic peak duration to systolic peak duration at 10% amplitude of systolic amplitude
    DW_SW_25: The ratio of diastolic peak duration to systolic peak duration at 25% amplitude of systolic amplitude
    DW_SW_33: The ratio of diastolic peak duration to systolic peak duration at 33% amplitude of systolic amplitude
    DW_SW_50: The ratio of diastolic peak duration to systolic peak duration at 50% amplitude of systolic amplitude
    DW_SW_66: The ratio of diastolic peak duration to systolic peak duration at 66% amplitude of systolic amplitude
    DW_SW_75: The ratio of diastolic peak duration to systolic peak duration at 75% amplitude of systolic amplitude
    PR_mean: Mean pulse rate

    Segment-based features:
    zcr: Zero crossing rate
    snr: Signal to noise ratio

    Args:
        sig (ArrayLike): Signal to be analyzed.
        fs (float): Sampling rate
        type (str, optional): Type of feature calculation, should be 'segment' or 'cycle'. Defaults to None.
        prefix (str, optional): Prefix for signal type. Defaults to 'signal'.

    Raises:
        ValueError: if Type is not 'cycle' or 'segment'.

    Returns:
        dict: Dictionary of calculated features.
    """

    if type=='cycle':
        features_time={}
        for key,func in FEATURES_TIME_CYCLE.items():
            features_time["_".join([prefix, key])]=func(sig,kwargs['peaks_amp'],kwargs['peaks_locs'],kwargs['troughs_locs'],fs)

    elif type=='segment':
        features_time={}
        for key,func in FEATURES_TIME_SEGMENT.items():
            features_time["_".join([prefix, key])]=func(sig,fs)

    else: 
        raise ValueError("Type should be 'cycle' or 'segment'.")

    return features_time


def calculate_SW(sig: ArrayLike, peaks: ArrayLike, locs_peaks: ArrayLike, locs_onsets: ArrayLike, fs: float, ratio: float) -> float:
    """Calculates systolic phase duration of the PPG waveform.

    Args:
        sig (ArrayLike): Signal
        peaks (ArrayLike): Peak amplitudes
        locs_peaks (ArrayLike): Peak locations
        locs_onsets (ArrayLike): Trough amplitudes
        fs (float): Sampling rate
        ratio (float): Ratio (signal amplitude/peak amplitude) at which the feature is calculated

    Returns:
        float: Mean systolic phase duration
    """

    SWs=[]
    for c in range(len(locs_onsets)-1):
        ind=np.where(sig[locs_onsets[c]:locs_peaks[c]] >= (ratio*peaks[c]))
        SWs.append(len(ind[0])/fs)

    return np.mean(SWs)


def calculate_DW(sig: ArrayLike, peaks: ArrayLike, locs_peaks: ArrayLike, locs_onsets: ArrayLike, fs: float, ratio: float) -> float:
    """Calcualtes diastolic phase duration of the waveform.

    Args:
        sig (ArrayLike): Signal
        peaks (ArrayLike): Peak amplitudes
        locs_peaks (ArrayLike): Peak locations
        locs_onsets (ArrayLike): Trough amplitudes
        fs (float): Sampling rate
        ratio (float): Ratio (signal amplitude/peak amplitude) at which the feature is calculated

    Returns:
        float: Mean diastolic phase duration
    """

    DWs=[]
    for c in range(len(locs_onsets)-1):
        ind=np.where(sig[locs_peaks[c]:locs_onsets[c+1]] >= (ratio*peaks[c]))
        DWs.append(len(ind[0])/fs)

    return np.mean(DWs)


def calculate_zcr(sig: ArrayLike) -> float:
    """Calculates zero crossing rate, defined as number of zero-crossings to signal length

    Args:
        sig (ArrayLike): Signal

    Returns:
        float: Zero crossing rate
    """

    sig_=sig-np.mean(sig)
    numZeroCrossing = len(np.where(np.diff(np.sign(sig_)))[0])

    return numZeroCrossing/len(sig_)


def calculate_snr(sig: ArrayLike) -> float:
    """Calculates signal to noise ratio.

    Args:
        sig (ArrayLike): Signal

    Returns:
        float: Signal to noise ratio
    """
    mn_sig = np.mean(sig)
    std_sig= np.std(sig)
    snratio=np.where(std_sig == 0, 0, mn_sig/std_sig).item()

    return snratio



