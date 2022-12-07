from scipy import signal
from numpy.typing import ArrayLike
import warnings

def filter_acc(sig: ArrayLike, sampling_rate: float, method: str='lowpass') -> ArrayLike:
    """Filters ACC signal using predefined filters.

    Args:
        sig (ArrayLike): Signal to be filtered.
        sampling_rate (float): Sampling rate of the signal (Hz).
        method (str, optional): Filtering method. Defaults to 'lowpass'.

    Raises:
        ValueError: If filtering method is not 'lowpass'
        ValueError: If sampling rate is not greater than zero.

    Returns:
        ArrayLike: Filtered ACC signal.
    """

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    method = method.lower()

    if method == 'lowpass': 
        filtered_sig = _filter_acc_lowpass(sig=sig, sampling_rate=sampling_rate)

    else:
        raise ValueError(f"Undefined method: {method}.")

    return filtered_sig


def _filter_acc_lowpass(sig: ArrayLike, sampling_rate: float) -> ArrayLike:

    N = 2
    btype = 'lowpass'
    W2 = 10 / (sampling_rate/2)
    warnings.warn(f"Default parameters will be used for filtering. {N}th order lowpass filter with f2=10 Hz.")

    sos = signal.butter(N, W2, btype, output='sos')
    filtered_sig=signal.sosfiltfilt(sos, sig)

    return filtered_sig
