from ..signaltools import filtering
from numpy.typing import ArrayLike

def filter_ppg(sig: ArrayLike, fs: float) -> ArrayLike:
    """Filters the ppg signal using predefined filter parameters

    Args:
        sig (array): PPG signal
        fs (float): Sampling rate

    Returns:
        (array): Filtered signal
    """
    N=2
    filter_type='bandpass'
    f1=0.5
    f2=5

    filtered_signal=filtering.filter_signal(sig,filter_type,N,fs,f1,f2)

    return filtered_signal