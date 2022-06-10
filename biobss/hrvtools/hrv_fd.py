import numpy as np
from scipy import fft 
from scipy import signal
import math
from numpy.typing import ArrayLike

#Frequency domain features
F_ULF=[0, 0.0033]
F_VLF=[0.0033, 0.04]
F_LF=[0.04, 0.15]
F_HF=[0.15, 0.4]
F_VHF=[0.4, 0.5]

FEATURES_FREQ= {
'ulf': lambda sig,fs:_pow(sig,fs,F_ULF),
'vlf': lambda sig,fs: _pow(sig,fs,F_VLF),
'lf': lambda sig,fs: _pow(sig,fs,F_LF),
'hf': lambda sig,fs: _pow(sig,fs,F_HF),
'vhf': lambda sig,fs: _pow(sig,fs,F_VHF),
'lf_hf_ratio': lambda sig,fs: _pow(sig,fs,F_LF)/_pow(sig,fs,F_HF),
'total_power': lambda sig,fs: _pow(sig,fs,F_VLF)+_pow(sig,fs,F_LF)+_pow(sig,fs,F_HF),
'lfnu': lambda sig,fs: (_pow(sig,fs,F_LF)/(_pow(sig,fs,F_LF)+_pow(sig,fs,F_HF)))*100,
'hfnu': lambda sig,fs: (_pow(sig,fs,F_HF)/(_pow(sig,fs,F_LF)+_pow(sig,fs,F_HF)))*100,
'lnHF': lambda sig,fs: np.log(_pow(sig,fs,F_HF)),
}  


def get_freq_features(sig: ArrayLike,ppi,fs: float, prefix: str='hrv') -> dict:
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

    features_freq={}

    for key,func in FEATURES_FREQ.items():
        features_freq["_".join([prefix, key])]=func(sig,fs)


    return features_freq


def _pow(sig,fs,freq_band):

    #nfft=2**math.ceil(math.log2(abs(len(sig))))
    nfft=len(sig) 
        
    freq=fft.fftfreq(nfft,1/fs)
    sigfft=np.abs(fft.fft(sig,nfft)/len(sig))
    P1=sigfft[0:int(nfft/2)]
    P1[1:-1] = 2 * P1[1:-1]
    sigfft=P1
    freq=freq[0 : int(len(sig)/ 2)]

    sig_=sig-np.mean(sig)
    f,pxx=signal.welch(sig_, fs=fs, nfft=nfft)

    pow= np.trapz(y=pxx[np.logical_and(f >= freq_band[0], f < freq_band[1])], x=f[np.logical_and(f >= freq_band[0], f < freq_band[1])])
    return pow