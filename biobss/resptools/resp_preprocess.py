from numpy.typing import ArrayLike

from ..preprocess import signal_filter


def elim_vlf(sig: ArrayLike, sampling_rate: float) -> ArrayLike:
    """Eliminates very low frequencies prior to respiratory rate estimation procedure.
    The cutoff frequencies are determined considering the frequency range of the respiration.

    Args:
        sig (Array): Input signal (ECG or PPG).
        sampling_rate (float): Sampling frequency of the input signal (Hz).

    Returns:
        Array: Filtered signal.
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    N = 5
    filter_type = "highpass"
    f1 = 0.0665  # 4 bpm

    filtered_signal = signal_filter.filter_signal(
        sig=sig, filter_type=filter_type, N=N, sampling_rate=sampling_rate, f_lower=f1
    )

    return filtered_signal


def elim_vhf(sig: ArrayLike, sampling_rate: float) -> ArrayLike:
    """Eliminates very high frequencies prior to respiratory rate estimation procedure.
    The cutoff frequencies are determined considering the frequency range of the respiration.

    Args:
        sig (Array): Input signal (ECG or PPG).
        sampling_rate (float): Sampling frequency of the input signal (Hz).

    Returns:
        Array: Filtered signal.
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    N = 5
    filter_type = "lowpass"
    f2 = 0.5833  # 35 bpm

    filtered_signal = signal_filter.filter_signal(
        sig=sig, filter_type=filter_type, N=N, sampling_rate=sampling_rate, f_upper=f2
    )

    return filtered_signal
