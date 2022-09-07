from ..signaltools import filtering
from numpy.typing import ArrayLike


def elim_vlf(ppg_sig: ArrayLike, sampling_rate: float) -> ArrayLike:
    """Eliminates very low frequencies prior to respiratory rate estimation procedure. 
    The cutoff frequencies are determined considering the frequency range of the respiration.

    Args:
        ppg_sig (Array): PPG signal to be analyzed.
        sampling_rate (float): Sampling frequency of the PPG signal (Hz).

    Returns:
        Array: Filtered PPG signal.
    """
    N = 5
    filter_type = 'highpass'
    f1 = 0.0665  # 4 bpm

    filtered_signal = filtering.filter_signal(
        ppg_sig, filter_type, N, sampling_rate, f1, [])

    return filtered_signal

def elim_vhf(ppg_sig: ArrayLike, sampling_rate: float) -> ArrayLike:
    """Eliminates very high frequencies prior to respiratory rate estimation procedure. 
    The cutoff frequencies are determined considering the frequency range of the respiration.

    Args:
        ppg_sig (Array): PPG signal to be analyzed.
        sampling_rate (float): Sampling frequency of the PPG signal (Hz).

    Returns:
        Array: Filtered PPG signal.
    """
    N = 5
    filter_type = 'lowpass'
    f2 = 30

    filtered_signal = filtering.filter_signal(
        ppg_sig, filter_type, N, sampling_rate, [], f2)

    return filtered_signal
