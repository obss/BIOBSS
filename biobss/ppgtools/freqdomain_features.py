import numpy as np
from scipy import fft 
from scipy import signal
from numpy.typing import ArrayLike

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


def sig_fft(sig: ArrayLike, sampling_rate: int) -> tuple:
    """Calculates Fast Fourier Transform (FFT) of a signal.

    Args:
        sig (ArrayLike): The signal to be analyzed.
        sampling_rate (int): Sampling frequency of the signal (Hz).

    Returns:
        tuple: fft frequencies, fft amplitudes
    """
    nfft=len(sig)  
    freq=fft.fftfreq(nfft,1/sampling_rate)
    sigfft=np.abs(fft.fft(sig,nfft)/len(sig))
    P1=sigfft[0:int(nfft/2)]
    P1[1:-1] = 2 * P1[1:-1]
    sigfft=P1
    freq=freq[0 : int(len(sig)/ 2)]

    return freq, sigfft

def fft_peaks(sigfft: ArrayLike, freq:ArrayLike, peakno: int, loc:bool=False) -> float:
    """Detects peaks from the fft of the signal. Returns peak amplitudes or peak locations (frequencies).

    Args:
        sigfft (ArrayLike): fft array to be analyzed.
        freq (ArrayLike): frequencies of the fft array.
        peakno (int): Index of the peak to be returned, when sorted in descending order.
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

def sig_psd(sig: ArrayLike, sampling_rate: int, method: str='welch') -> tuple:
    """Calculate Power Spectral Density (PSD) of a signal using 'fft' or 'welch' method.

    Args:
        sig (ArrayLike): The signal to be analyzed.
        sampling_rate (int): Sampling frequency of the signal. 
        method (str, optional): Method to calculate PSD. It can be 'welch' or 'fft'. Defaults to 'welch'.

    Raises:
        ValueError: If 'method' is not one of 'fft' and 'welch'.

    Returns:
        tuple: psd frequencies, psd amplitudes
    """

    if method == 'welch':
        fxx, pxx = _sig_psd_welch(sig, sampling_rate=sampling_rate)

    elif method =='fft':
        fxx, pxx = _sig_psd_fft(sig, sampling_rate=sampling_rate)

    else:
        raise ValueError("Method should be 'fft' or 'welch'. ")

    return fxx, pxx

def sig_power(pxx:ArrayLike, fxx:ArrayLike, freq_range:list) -> float:
    """Calculates power of the signal for a given frequency range.

    Args:
        pxx (ArrayLike): Array of power spectral density values.
        fxx (ArrayLike): frequencies of the pxx array.
        freq_range (list): Frequency range to calculate signal power.

    Returns:
        float: Power of the signal for the given frequency range.
    """
    f1 = freq_range[0]
    f2=freq_range[1]
    pow=np.trapz(pxx[np.logical_and(fxx>=f1,fxx<f2)],fxx[np.logical_and(fxx>=f1,fxx<f2)])

    return pow

def _sig_psd_fft(sig, sampling_rate):

    freq, sigfft = sig_fft(sig, sampling_rate=sampling_rate)
    psd = (np.abs(sigfft) ** 2) / np.diff(freq)[0]

    return freq, psd

def _sig_psd_welch(sig, sampling_rate):
    nfft=len(sig)
    sig_ = sig - np.mean(sig)
    freq, psd = signal.welch(sig_, fs=sampling_rate, window='hann', nfft=nfft)

    return freq, psd

