import numpy as np
from scipy import fft 
from scipy import signal
from numpy.typing import ArrayLike

from biobss.common.signal_fft import *
from biobss.common.signal_power import *

#Frequency domain features
FUNCTIONS_FREQ_SEGMENT= {
'p_1': lambda sigfft,freq,_0,_1: fft_peaks(sigfft,freq,1),
'f_1': lambda sigfft,freq,_0,_1: fft_peaks(sigfft,freq,1,loc=True),
'p_2': lambda sigfft,freq,_0,_1: fft_peaks(sigfft,freq,2),
'f_2': lambda sigfft,freq,_0,_1: fft_peaks(sigfft,freq,2,loc=True),
'p_3': lambda sigfft,freq,_0,_1: fft_peaks(sigfft,freq,3),
'f_3': lambda sigfft,freq,_0,_1: fft_peaks(sigfft,freq,3,loc=True),
'pow': lambda _0,_1,pxx,fxx: sig_power(pxx,fxx,[0,2]),
'rpow': lambda _0,_1,pxx,fxx: sig_power(pxx,fxx,[0,2.25]) / sig_power(pxx,fxx,[0,5]),
}  


def get_freq_features(sig: ArrayLike, sampling_rate: float, type: str, prefix: str='signal') -> dict:
    """Calculates frequency-domain features

    Segment-based features:
    p_1: The amplitude of the first peak from the fft of the signal
    f_1: The frequency at which the first peak from the fft of the signal occurred
    p_2: The amplitude of the second peak from the fft of the signal
    f_2: The frequency at which the second peak from the fft of the signal occurred
    p_3: The amplitude of the third peak from the fft of the signal
    f_3: The frequency at which the third peak from the fft of the signal occurred
    pow: Power of the signal at a given range of frequencies
    rpow: Ratio of the powers of the signal at given ranges of frequencies

    Args:
        sig (ArrayLike): Signal
        sampling_rate (float): Sampling rate
        type (str): Type of feature calculation, should be 'segment'. Defaults to None.
        prefix (str, optional): Prefix for signal type. Defaults to 'signal'.

    Raises:
        ValueError: if type is not 'segment'.

    Returns:
        dict: Dictionary of calculated features
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    type = type.lower()
    
    if type=='segment':
    
        nfft=len(sig) 
         
        freq=fft.fftfreq(nfft,1/sampling_rate)
        sigfft=np.abs(fft.fft(sig,nfft)/len(sig))
        P1=sigfft[0:int(nfft/2)]
        P1[1:-1] = 2 * P1[1:-1]
        sigfft=P1
        freq=freq[0 : int(len(sig)/ 2)]

        sig_=sig-np.mean(sig)
        f,pxx=signal.welch(sig_, fs=sampling_rate, nfft=nfft)
        
        features_freq={}
        for key,func in FUNCTIONS_FREQ_SEGMENT.items():
            features_freq["_".join([prefix, key])]=func(sigfft,freq,pxx,f)

    else:
        raise ValueError("Undefined type for frequency domain.")

    return features_freq


