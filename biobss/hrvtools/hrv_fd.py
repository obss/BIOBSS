import numpy as np
from scipy import fft 
from scipy import signal
from scipy import interpolate
import math
from numpy.typing import ArrayLike

#Frequency domain features
F_ULF=[0, 0.0033]
F_VLF=[0.0033, 0.04]
F_LF=[0.04, 0.15]
F_HF=[0.15, 0.4]
F_VHF=[0.4, 0.5]

FEATURES_FREQ= {
'ulf': lambda pxx,f:_pow(pxx,f,F_ULF),
'vlf': lambda pxx,f: _pow(pxx,f,F_VLF),
'lf': lambda pxx,f: _pow(pxx,f,F_LF),
'hf': lambda pxx,f: _pow(pxx,f,F_HF),
'vhf': lambda pxx,f: _pow(pxx,f,F_VHF),
'lf_hf_ratio': lambda pxx,f: _pow(pxx,f,F_LF)/_pow(pxx,f,F_HF),
'total_power': lambda pxx,f: _pow(pxx,f,F_VLF)+_pow(pxx,f,F_LF)+_pow(pxx,f,F_HF),
'lfnu': lambda pxx,f: (_pow(pxx,f,F_LF)/(_pow(pxx,f,F_LF)+_pow(pxx,f,F_HF)))*100,
'hfnu': lambda pxx,f: (_pow(pxx,f,F_HF)/(_pow(pxx,f,F_LF)+_pow(pxx,f,F_HF)))*100,
'lnHF': lambda pxx,f: np.log(_pow(pxx,f,F_HF)),
}  


def hrv_freq_features(ppi, prefix: str='hrv') -> dict:
    """Calculates frequency-domain features
    hrv_ulf: The spectral power density pertaining to ultra low frequency band i.e., 0 to .0033 Hz by default.
    hrv_vlf: The spectral power density pertaining to ultra low frequency band i.e., 0.0033 to 0.04 Hz by default.
    hrv_lf: The spectral power density pertaining to ultra low frequency band i.e., 0.04 to 0.15 Hz by default.
    hrv_hf: The spectral power density pertaining to ultra low frequency band i.e., 0.15 to 0.4 Hz by default.
    hrv_vhf: The variability, or signal power, in very high frequency i.e., 0.4 to 0.5 Hz by default.
    LF/HF: the ratio of LF to HF
    hrv_lfnu: 
    hrv_hfnu:
    LnHF: The log transformed HF.

    
        - **LFn**: The normalized low frequency, obtained by dividing the low frequency power by
        the total power.
        - **HFn**: The normalized high frequency, obtained by dividing the low frequency power by
        the total power.
 

    Args:
        sig (ArrayLike): Signal
        fs (float): Sampling rate
        type (str): Type of feature calculation, should be 'segment'. Defaults to None.
        prefix (str, optional): Prefix for signal type. Defaults to 'signal'.

    Raises:
        ValueError: if type is not 'segment'.

    Returns:
        dict: Dictionary of calculated features
    """

    #Interpolate the ppi array
    f_interp=4
    t=np.zeros(len(ppi))

    for i in range(len(ppi)):
        print(i)
        t[i]=np.sum(ppi[:i+1])

    t2 = np.arange(t[0],t[-1]+1/f_interp,1/f_interp)

    interp=interpolate.CubicSpline(t,ppi)
    y=interp(t2)

    #Calculate power spectral density using Welch method
    y1=y-np.mean(y)
    #nfft=2**math.ceil(math.log2(abs(len(y1))))
    nfft=len(y1)
    f,pxx=signal.welch(y1, fs=f_interp, nfft=nfft)

    #freq=fft.fftfreq(nfft,1/f_interp)
    #sigfft=np.abs(fft.fft(y1,nfft)/len(y1))
    #P1=sigfft[0:int(nfft/2)]
    #P1[1:-1] = 2 * P1[1:-1]
    #sigfft=P1
    #freq=freq[0 : int(len(y1)/ 2)]

    features_freq={}

    for key,func in FEATURES_FREQ.items():
        features_freq["_".join([prefix, key])]=func(pxx,f)


    return features_freq


def _pow(pxx,f,freq_band):

    pow= np.trapz(y=pxx[np.logical_and(f >= freq_band[0], f < freq_band[1])], x=f[np.logical_and(f >= freq_band[0], f < freq_band[1])])
    return pow