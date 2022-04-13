import numpy as np
from scipy import fft 
from scipy import signal
import math
from numpy.typing import ArrayLike

#Frequency domain features
FUNCTIONS_FREQ_SEGMENT= {
'p_1': lambda sigfft,freq,_0,_1: fft_peaks(sigfft,freq,1),
'f_1': lambda sigfft,freq,_0,_1: fft_peaks(sigfft,freq,1,True),
'p_2': lambda sigfft,freq,_0,_1: fft_peaks(sigfft,freq,2),
'f_2': lambda sigfft,freq,_0,_1: fft_peaks(sigfft,freq,2,True),
'p_3': lambda sigfft,freq,_0,_1: fft_peaks(sigfft,freq,3),
'f_3': lambda sigfft,freq,_0,_1: fft_peaks(sigfft,freq,3,True),
'pow': lambda _0,_1,pxx,f: fft_pow(pxx,f,0,2),
'rpow': lambda _0,_1,pxx,f: fft_relpow(pxx,f,[0,2.25],[0,5]),
}  


def get_freq_features(sig: ArrayLike, fs: float, type: str, prefix: str='signal') -> dict:
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
        fs (float): Sampling rate
        type (str): Type of feature calculation, should be 'segment'. Defaults to None.
        prefix (str, optional): Prefix for signal type. Defaults to 'signal'.

    Raises:
        ValueError: if type is not equal to 'segment'.

    Returns:
        dict: Dictionary of calculated features
    """

    if type=='segment':

        #nfft=2**math.ceil(math.log2(abs(len(sig))))
        nfft=len(sig)
            
        freq=fft.fftfreq(nfft,1/fs)
        sigfft=np.abs(fft.fft(sig,nfft)/len(sig))
        P1=sigfft[0:nfft/2]
        P1[1:-1] = 2 * P1[1:-1]
        sigfft=P1
        freq=freq[0 : int(len(sig)/ 2)]

        sig_=sig-np.mean(sig)
        #n=math.ceil(math.log2(abs(len(sig_))))
        f,pxx=signal.welch(sig_, fs=fs, nfft=nfft)
        
        features_freq={}
        for key,func in FUNCTIONS_FREQ_SEGMENT.items():
            features_freq["_".join([prefix, key])]=func(sigfft,freq,pxx,f)

    else:
        raise ValueError("Undefined type for frequency domain.")

    return features_freq


def fft_peaks(sigfft: ArrayLike, freq:ArrayLike, peakno: int, loc:bool=False) -> float:
    """Detects peaks from the fft of the signal. Returns peak amplitudes or peak locations (frequencies).

    Args:
        sigfft (ArrayLike): fft array to be analyzed.
        freq (ArrayLike): frequencies of the fft array.
        peakno (int): Number of the peak to be returned, when sorted in descending order.
        loc (bool, optional): If True, frequency value is returned. Defaults to False.

    Returns:
        float: Amplitude of the peak or frequency at which peak occurred
    """
    
    locs_fft, _=signal.find_peaks(sigfft)
    peaks_fft=sigfft[locs_fft]
    freq_fft=freq[locs_fft]

    sorted_ind=(-peaks_fft).argsort()
    sorted_peaks=peaks_fft[sorted_ind]
    sorted_freq=freq_fft[sorted_ind]
    
    if not loc:
        return sorted_peaks[peakno-1]

    else:
        return sorted_freq[peakno-1]


def fft_pow(pxx:ArrayLike, f:ArrayLike, f1:float, f2:float) -> float:
    """Calculates power of the signal for a given frequency range.

    Args:
        pxx (ArrayLike): Array of power spectral density values.
        f (ArrayLike): frequencies of the pxx array.
        f1 (float): Lower limit of the frequency range.
        f2 (float): Upper limit of the frequency range.

    Returns:
        float: Power of the signal for the given frequency range.
    """

    pow=np.trapz(pxx[np.logical_and(f>=f1,f<=f2)],f[np.logical_and(f>=f1,f<=f2)])

    return pow

def fft_relpow(pxx,f,F1,F2) -> float:
    """Calculates power of the signal for the given frequency ranges.

    Args:
        pxx (_type_): Array of power spectral density values.
        f (_type_): frequencies of the pxx array.
        F1 (_type_): Lower limit of the frequency range.
        F2 (_type_): Upper limit of the frequency range.

    Returns:
        float: Relative power of the signal for the given frequency ranges.
    """

    powerF1 = np.trapz(pxx[np.logical_and(f>=F1[0],f<=F1[1])],f[np.logical_and(f>=F1[0],f<=F1[1])])
    powerF2 = np.trapz(pxx[np.logical_and(f>=F2[0],f<=F2[1])],f[np.logical_and(f>=F2[0],f<=F2[1])])   

    return powerF1/powerF2
