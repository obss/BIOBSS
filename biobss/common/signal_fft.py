import numpy as np
from numpy.typing import ArrayLike
from scipy import fft, signal


def sig_fft(sig: ArrayLike, sampling_rate: float) -> tuple:
    """Calculates Fast Fourier Transform (FFT) of a signal.

    Args:
        sig (ArrayLike): Input signal.
        sampling_rate (float): Sampling frequency of the signal (Hz).

    Returns:
        tuple: FFT frequencies, FFT amplitudes
    """
    nfft = len(sig)
    freq = fft.fftfreq(nfft, 1 / sampling_rate)
    sigfft = np.abs(fft.fft(sig, nfft) / len(sig))
    P1 = sigfft[0 : int(nfft / 2)]
    P1[1:-1] = 2 * P1[1:-1]
    sigfft = P1
    freq = freq[0 : int(len(sig) / 2)]

    return freq, sigfft


def fft_peaks(sigfft: ArrayLike, freq: ArrayLike, peakno: int, loc: bool = False) -> float:
    """Detects peaks from the FFT of the signal and returns the highest Mth (peakno) peak amplitude or peak location (frequency).

    Args:
        sigfft (ArrayLike): Array of FFT amplitudes
        freq (ArrayLike): Array of FFT frequencies
        peakno (int): Index of the peak to be returned, when sorted in descending order.
        loc (bool, optional): If True, FFT frequency is returned. Defaults to False.

    Returns:
        float: Amplitude or location of the peak.
    """
    locs_fft, _ = signal.find_peaks(sigfft)
    peaks_fft = sigfft[locs_fft]
    freq_fft = freq[locs_fft]

    sorted_ind = (-peaks_fft).argsort()
    sorted_peaks = peaks_fft[sorted_ind]
    sorted_freq = freq_fft[sorted_ind]

    if not loc:
        return sorted_peaks[peakno - 1]
    else:
        return sorted_freq[peakno - 1]
