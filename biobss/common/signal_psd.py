import numpy as np
from scipy import fft 
from scipy import signal
from numpy.typing import ArrayLike

from biobss.common.signal_fft import *

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

def _sig_psd_fft(sig, sampling_rate):

    freq, sigfft = sig_fft(sig, sampling_rate=sampling_rate)
    psd = (np.abs(sigfft) ** 2) / np.diff(freq)[0]

    return freq, psd

def _sig_psd_welch(sig, sampling_rate):
    nfft=len(sig)
    sig_ = sig - np.mean(sig)
    freq, psd = signal.welch(sig_, fs=sampling_rate, window='hann', nfft=nfft)

    return freq, psd