import numpy as np
from numpy.typing import ArrayLike

#Time domain features
FEATURES_TIME_CYCLE = {
'a_S': lambda _0,peaks_amp,_1,_2,_3,_4: np.mean(peaks_amp),
't_S': lambda _0,_1,peaks_locs,_2,troughs_locs,sampling_rate: np.mean((peaks_locs-troughs_locs[:-1])/sampling_rate),
't_C': lambda _0,_1,_2,_3, troughs_locs,sampling_rate: np.mean(np.diff(troughs_locs)/sampling_rate),
'DW': lambda _0,_1,peaks_locs,_2,troughs_locs,sampling_rate: np.mean((troughs_locs[1:]-peaks_locs)/sampling_rate),
'SW_10': lambda sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate: _calculate_SW(sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate,0.1),
'SW_25': lambda sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate: _calculate_SW(sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate,0.25),
'SW_33': lambda sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate: _calculate_SW(sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate,0.33),
'SW_50': lambda sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate: _calculate_SW(sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate,0.5),
'SW_66': lambda sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate: _calculate_SW(sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate,0.66),
'SW_75': lambda sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate: _calculate_SW(sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate,0.75),
'DW_10': lambda sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate: _calculate_DW(sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate,0.1),
'DW_25': lambda sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate: _calculate_DW(sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate,0.25),
'DW_33': lambda sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate: _calculate_DW(sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate,0.33),
'DW_50': lambda sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate: _calculate_DW(sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate,0.5),
'DW_66': lambda sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate: _calculate_DW(sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate,0.66),
'DW_75': lambda sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate: _calculate_DW(sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate,0.75),
'DW_SW_10': lambda sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate: _calculate_DW_SW(sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate,0.1),
'DW_SW_25': lambda sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate: _calculate_DW_SW(sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate,0.25),
'DW_SW_33': lambda sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate: _calculate_DW_SW(sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate,0.33),
'DW_SW_50': lambda sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate: _calculate_DW_SW(sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate,0.50),
'DW_SW_66': lambda sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate: _calculate_DW_SW(sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate,0.66),
'DW_SW_75': lambda sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate: _calculate_DW_SW(sig,peaks_amp,peaks_locs,troughs_amp,troughs_locs,sampling_rate,0.75),
'PR_mean' : lambda _0,_1,peaks_locs,_2,_3,sampling_rate: 60/np.mean(np.diff(peaks_locs)/sampling_rate),
}

FEATURES_TIME_SEGMENT = {
'zcr': lambda sig,_0: _calculate_zcr(sig),
'snr': lambda sig,_0: _calculate_snr(sig),
}

def ppg_time_features(sig: ArrayLike, sampling_rate: float, input_types: str, prefix: str='ppg', **kwargs) -> dict:
    """Calculates time-domain features.

    Cycle-based features:
    a_S: Mean amplitude of the systolic peaks 
    t_S: Mean systolic peak duration
    t_C: Mean cycle duration
    DW: Mean diastolic peak duration
    SW_10: The systolic peak duration at 10% of systolic amplitude
    SW_25: The systolic peak duration at 25% of systolic amplitude
    SW_33: The systolic peak duration at 33% of systolic amplitude
    SW_50: The systolic peak duration at 50% of systolic amplitude
    SW_66: The systolic peak duration at 66% of systolic amplitude
    SW_75: The systolic peak duration at 75% of systolic amplitude
    DW_10: The diastolic peak duration at 10% of systolic amplitude
    DW_25: The diastolic peak duration at 25% of systolic amplitude
    DW_33: The diastolic peak duration at 33% of systolic amplitude
    DW_50: The diastolic peak duration at 50% of systolic amplitude
    DW_66: The diastolic peak duration at 66% of systolic amplitude
    DW_75: The diastolic peak duration at 75% of systolic amplitude
    DW_SW_10: The ratio of diastolic peak duration to systolic peak duration at 10% of systolic amplitude
    DW_SW_25: The ratio of diastolic peak duration to systolic peak duration at 25% of systolic amplitude
    DW_SW_33: The ratio of diastolic peak duration to systolic peak duration at 33% of systolic amplitude
    DW_SW_50: The ratio of diastolic peak duration to systolic peak duration at 50% of systolic amplitude
    DW_SW_66: The ratio of diastolic peak duration to systolic peak duration at 66% of systolic amplitude
    DW_SW_75: The ratio of diastolic peak duration to systolic peak duration at 75% of systolic amplitude
    PR_mean: Mean pulse rate

    Segment-based features:
    zcr: Zero crossing rate
    snr: Signal to noise ratio

    Args:
        sig (ArrayLike): Signal to be analyzed.
        sampling_rate (float): Sampling rate
        type (str, optional): Type of feature calculation, should be 'segment' or 'cycle'. Defaults to None.
        prefix (str, optional): Prefix for signal type. Defaults to 'signal'.

    Kwargs:
        peaks_amp (ArrayLike): Array of peak amplitudes
        peaks_locs (ArrayLike): Array of peak locations
        troughs_amp (ArrayLike): Array of trough amplitudes
        troughs_locs (ArrayLike): Array of trough locations

    Raises:
        ValueError: if Type is not 'cycle' or 'segment'.

    Returns:
        dict: Dictionary of calculated features.
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    input_types = [x.lower() for x in input_types]

    features_time={}
    for type in input_types:
        if type=='cycle':            
            for key,func in FEATURES_TIME_CYCLE.items():
                features_time["_".join([prefix, key])]=func(sig, kwargs['peaks_amp'], kwargs['peaks_locs'], kwargs['troughs_amp'], kwargs['troughs_locs'],sampling_rate)

        elif type=='segment':
            for key,func in FEATURES_TIME_SEGMENT.items():
                features_time["_".join([prefix, key])]=func(sig,sampling_rate)

        else: 
            raise ValueError("Type should be 'cycle' or 'segment'.")

    return features_time

def _calculate_SW(sig: ArrayLike, peaks_amp: ArrayLike, peaks_locs: ArrayLike, troughs_amp: ArrayLike, troughs_locs: ArrayLike, sampling_rate: float, ratio: float) -> float:
    """Calculates systolic phase duration of the PPG waveform."""
    
    SWs=[]
    for c in range(len(troughs_locs)-1):
        sys_amp=peaks_amp[c]-troughs_amp[c]
        thresh=(ratio*sys_amp)+troughs_amp[c]
        ind=np.where(sig[troughs_locs[c]:peaks_locs[c]] >= thresh)
        SWs.append(len(ind[0])/sampling_rate)

    return np.mean(SWs)

def _calculate_DW(sig: ArrayLike, peaks_amp: ArrayLike, peaks_locs: ArrayLike, troughs_amp: ArrayLike, troughs_locs: ArrayLike, sampling_rate: float, ratio: float) -> float:
    """Calculates diastolic phase duration of the waveform."""

    DWs=[]
    for c in range(len(troughs_locs)-1):
        sys_amp=peaks_amp[c]-troughs_amp[c]
        thresh=(ratio*sys_amp)+troughs_amp[c]
        ind=np.where(sig[peaks_locs[c]:troughs_locs[c+1]] >= thresh)
        DWs.append(len(ind[0])/sampling_rate)

    return np.mean(DWs)

def _calculate_DW_SW(sig: ArrayLike, peaks_amp: ArrayLike, peaks_locs: ArrayLike, troughs_amp: ArrayLike, troughs_locs: ArrayLike, sampling_rate: float, ratio: float) -> float:
    """Calculates the ratio of diastolic phase duration of the waveform to systolic phase duration of the waveform."""

    dw = _calculate_DW(sig, peaks_amp=peaks_amp, peaks_locs=peaks_locs, troughs_amp=troughs_amp, troughs_locs=troughs_locs, sampling_rate=sampling_rate, ratio=ratio)
    sw = _calculate_SW(sig, peaks_amp=peaks_amp, peaks_locs=peaks_locs, troughs_amp=troughs_amp, troughs_locs=troughs_locs, sampling_rate=sampling_rate, ratio=ratio)

    return dw / sw 

def _calculate_zcr(sig: ArrayLike) -> float:
    """Calculates zero crossing rate, defined as number of zero-crossings to signal length."""

    sig_=sig-np.mean(sig)
    numZeroCrossing = len(np.where(np.diff(np.sign(sig_)))[0])

    return numZeroCrossing/len(sig_)

def _calculate_snr(sig: ArrayLike) -> float:
    """Calculates signal to noise ratio."""

    mn_sig = np.mean(sig)
    std_sig= np.std(sig)
    snratio=np.where(std_sig == 0, 0, mn_sig/std_sig).item()

    return snratio



