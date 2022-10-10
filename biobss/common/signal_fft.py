import numpy as np
from scipy import fft, signal
from numpy.typing import ArrayLike


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