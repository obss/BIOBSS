import numpy as np
from numpy.typing import ArrayLike
from scipy import signal

from biobss.common.signal_fft import *


def sig_psd(sig: ArrayLike, sampling_rate: float, method: str = "welch") -> tuple:
    """Calculates Power Spectral Density (PSD) of a signal using 'fft' or 'welch' method.

    Args:
        sig (ArrayLike): Input signal.
        sampling_rate (float): Sampling rate of the signal (Hz).
        method (str, optional): Method to calculate Power Spectral Density(PSD). It can be 'welch' or 'fft'. Defaults to 'welch'.

    Raises:
        ValueError: If 'method' is not one 'fft' or 'welch'.

    Returns:
        tuple: PSD frequencies, PSD amplitudes
    """

    if method == "welch":
        fxx, pxx = _sig_psd_welch(sig, sampling_rate=sampling_rate)

    elif method == "fft":
        fxx, pxx = _sig_psd_fft(sig, sampling_rate=sampling_rate)

    else:
        raise ValueError("Method should be 'fft' or 'welch'. ")

    return fxx, pxx


def sig_power(pxx: ArrayLike, fxx: ArrayLike, freq_range: list) -> float:
    """Calculates signal power from power spectral density for a given frequency range.

    Args:
        pxx (ArrayLike): Array of power spectral density values.
        fxx (ArrayLike): Frequencies corresponding to pxx array.
        freq_range (list): Frequency range to calculate signal power.

    Returns:
        float: Power of the signal for the given frequency range.
    """
    f1 = freq_range[0]
    f2 = freq_range[1]
    pow = np.trapz(pxx[np.logical_and(fxx >= f1, fxx < f2)], fxx[np.logical_and(fxx >= f1, fxx < f2)])

    return pow


def _sig_psd_fft(sig, sampling_rate):

    sig_ = sig - np.mean(sig)
    freq, sigfft = sig_fft(sig_, sampling_rate=sampling_rate)
    psd = (np.abs(sigfft) ** 2) / np.diff(freq)[0]

    return freq, psd


def _sig_psd_welch(sig, sampling_rate):

    nfft = len(sig)
    sig_ = sig - np.mean(sig)
    freq, psd = signal.welch(sig_, fs=sampling_rate, window="hann", nfft=nfft)

    return freq, psd
