from ..signaltools import filtering
from numpy.typing import ArrayLike
from ..pipeline.signal import Signal

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


def filter_ppg_signal(signal: Signal) -> Signal:
    """Filters the ppg signal using predefined filter parameters

    Args:
        signal (Signal): PPG signal

    Returns:
        (Signal): Filtered signal
    """

        
    filtered_signal=filter_ppg(signal.signal,signal.sampling_rate)
    filtered_signal=Signal(filtered_signal,signal.sampling_rate,signal.modality,signal.signal_name)

    return filtered_signal

