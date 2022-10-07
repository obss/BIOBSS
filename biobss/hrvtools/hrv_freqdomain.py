import numpy as np
from scipy import fft 
from scipy import signal
from scipy import interpolate
from scipy import integrate
import math
from numpy.typing import ArrayLike

from biobss.common.signal_fft import *
from biobss.common.signal_power import *
from biobss.common.signal_psd import *

F_INTERP = 4
F_VLF=[0, 0.04]
F_LF=[0.04, 0.15]
F_HF=[0.15, 0.4]

#Frequency domain features
FEATURES_FREQ= {
'vlf': lambda pxx,fxx: sig_power(pxx,fxx,F_VLF),
'lf': lambda pxx,fxx: sig_power(pxx,fxx,F_LF),
'hf': lambda pxx,fxx: sig_power(pxx,fxx,F_HF),
'lf_hf_ratio': lambda pxx,fxx: sig_power(pxx,fxx,F_LF)/sig_power(pxx,fxx,F_HF),
'total_power': lambda pxx,fxx: sig_power(pxx,fxx,F_VLF)+sig_power(pxx,fxx,F_LF)+sig_power(pxx,fxx,F_HF),
'lfnu': lambda pxx,fxx: (sig_power(pxx,fxx,F_LF)/(sig_power(pxx,fxx,F_LF)+sig_power(pxx,fxx,F_HF)))*100,
'hfnu': lambda pxx,fxx: (sig_power(pxx,fxx,F_HF)/(sig_power(pxx,fxx,F_LF)+sig_power(pxx,fxx,F_HF)))*100,
'lnLF': lambda pxx,fxx: np.log(sig_power(pxx,fxx,F_LF)),
'lnHF': lambda pxx,fxx: np.log(sig_power(pxx,fxx,F_HF)),
'vlf_peak': lambda pxx,fxx: _peak_psd(pxx,fxx,F_VLF),
'lf_peak': lambda pxx,fxx: _peak_psd(pxx,fxx,F_LF),
'hf_peak': lambda pxx,fxx: _peak_psd(pxx,fxx,F_HF),
}  

def hrv_freq_features(ppi: ArrayLike, sampling_rate:int, prefix: str='hrv') -> dict:
    """Calculates frequency-domain hrv parameters.

    vlf: The spectral power pertaining to very low frequency band i.e., 0.0033 to 0.04 Hz by default.
    lf: The spectral power pertaining to low frequency band i.e., 0.04 to 0.15 Hz by default.
    hf: The spectral power pertaining to high frequency band i.e., 0.15 to 0.4 Hz by default.
    lf_hf_ratio: the ratio of LF to HF
    total_power: The sum of spectral powers pertaining to low and high frequency bands.
    lfnu: The normalized spectral power pertaining to low frequency band, obtained by dividing the low frequency power by the total power.
    hfnu: The normalized spectral power pertaining to low frequency band, obtained by dividing the high frequency power by the total power.
    lnLF: The log transformed low frequency power.
    lnHF: The log transformed high frequency power.

    Args:
        ppi (ArrayLike): Peak-to-peak interval array (miliseconds).
        prefix (str, optional): Prefix for the calculated parameters. Defaults to 'hrv'.

    Returns:
        dict: Dictionary of frequency-domain hrv parameters.
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")
        
    #Interpolate the ppi array
    t=np.cumsum(ppi)/1000
    interp=interpolate.interp1d(t, ppi, kind='cubic', fill_value="extrapolate")
    steps = 1 / F_INTERP
    t2 = np.arange(t[0], t[-1]+steps, steps)
    y=interp(t2)
    
    #Calculate power spectral density using Welch method
    fxx,pxx=sig_psd(y, sampling_rate=F_INTERP, method='welch')
    
    features_freq={}
    for key,func in FEATURES_FREQ.items():
        features_freq["_".join([prefix, key])]=func(pxx,fxx)

    return features_freq


def _peak_psd(pxx,f,freq_band):

    peak_freq = f[np.logical_and(f >= freq_band[0], f < freq_band[1])][np.argmax(pxx[np.logical_and(f >= freq_band[0], f < freq_band[1])])]

    return peak_freq