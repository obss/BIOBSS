from numpy.typing import ArrayLike
from scipy import signal


def filter_acc(sig: ArrayLike, sampling_rate: float, method: str = "lowpass") -> ArrayLike:
    """Filters ACC signal using predefined filter parameters.

    Args:
        sig (ArrayLike): ACC signal.
        sampling_rate (float): Sampling rate of the ACC signal (Hz).
        method (str, optional): Filtering method. Defaults to 'lowpass'.

    Raises:
        ValueError: If filtering method is not 'lowpass'.
        ValueError: If sampling rate is not greater than 0.

    Returns:
        ArrayLike: Filtered ACC signal.
    """

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    method = method.lower()

    if method == "lowpass":
        filtered_sig = _filter_acc_lowpass(sig=sig, sampling_rate=sampling_rate)

    else:
        raise ValueError(f"Undefined method: {method}.")

    return filtered_sig


def _filter_acc_lowpass(sig: ArrayLike, sampling_rate: float) -> ArrayLike:
    """Filters ACC signal using a predefined lowpass filter."""
    N = 2
    btype = "lowpass"
    W2 = 10 / (sampling_rate / 2)

    sos = signal.butter(N, W2, btype, output="sos")
    filtered_sig = signal.sosfiltfilt(sos, sig)

    return filtered_sig
