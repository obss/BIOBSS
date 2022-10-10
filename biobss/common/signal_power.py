import numpy as np
from scipy import fft 
from scipy import signal
from numpy.typing import ArrayLike

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