import warnings

from numpy.typing import ArrayLike
from scipy import signal


def filter_ppg(sig: ArrayLike, sampling_rate: float, method: str = "bandpass") -> ArrayLike:
    """Filters PPG signal using predefined filters.

    Args:
        sig (ArrayLike): PPG signal to be filtered.
        sampling_rate (float): Sampling rate of the PPG signal.
        method (str, optional): Filtering method. Defaults to 'bandpass'.

    Raises:
        ValueError: If sampling rate is not greater than zero.
        ValueError: If method is undefined.

    Returns:
        ArrayLike: Filtered PPG signal.
    """

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    method = method.lower()

    if method == "bandpass":
        filtered_sig = _filter_ppg_bandpass(sig=sig, sampling_rate=sampling_rate)

    else:
        raise ValueError(f"Undefined method: {method}.")

    return filtered_sig


def _filter_ppg_bandpass(sig: ArrayLike, sampling_rate: float) -> ArrayLike:

    N = 2
    btype = "bandpass"
    W1 = 0.5 / (sampling_rate / 2)
    W2 = 5 / (sampling_rate / 2)
    warnings.warn(
        f"Default parameters will be used for filtering. {N}th order bandpass filter with f1=0.5 Hz and f2=5 Hz."
    )

    sos = signal.butter(N, [W1, W2], btype, output="sos")
    filtered_sig = signal.sosfiltfilt(sos, sig)

    return filtered_sig
