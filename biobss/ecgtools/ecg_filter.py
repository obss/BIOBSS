from numpy.typing import ArrayLike
from scipy import signal


def filter_ecg(sig: ArrayLike, sampling_rate: float, method: str, **kwargs) -> ArrayLike:
    """Filters ECG signal using predefined filter parameters.

    Args:
            sig (ArrayLike): ECG signal.
            sampling_rate (float): Sampling rate of the ECG signal (Hz).
            method (str): Filtering method. Should be one of ['notch', 'bandpass', 'pantompkins', 'hamilton', 'elgendi].

    Kwargs:
            f_notch (float) : Center frequency of the notch filter (w0).
            quality_factor (float): Quality factor (Q). It is calculated as Q = w0/bw where bw is the -3dB bandwidth.

    Raises:
            ValueError: If sampling rate is less than or equal to 0.
            ValueError: If cut-off frequency is less than 0.
            ValueError: If required parameters are not provided for the selected method.
            ValueError: If filtering method is not one of ['notch', 'pantompkins', 'hamilton', 'elgendi].

    Returns:
            ArrayLike: Filtered ECG signal.
    """

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    method = method.lower()

    if method == "notch":
        filtered_sig = _filter_ecg_notch(sig=sig, sampling_rate=sampling_rate, **kwargs)
    elif method == "pantompkins":
        filtered_sig = _filter_ecg_pantompkins(sig=sig, sampling_rate=sampling_rate)
    elif method == "hamilton":
        filtered_sig = _filter_ecg_hamilton(sig=sig, sampling_rate=sampling_rate)
    elif method == "elgendi":
        filtered_sig = _filter_ecg_elgendi(sig=sig, sampling_rate=sampling_rate)
    else:
        raise ValueError(f"Undefined method: {method}.")

    return filtered_sig


def _filter_ecg_notch(sig: ArrayLike, sampling_rate: float, **kwargs) -> ArrayLike:
    """Filters ECG signal using a Notch filter."""

    if all(k in kwargs.keys() for k in ("f_notch", "quality_factor")):
        if kwargs["f_notch"] <= 0:
            raise ValueError("Cut-off frequencies must be greater than 0.")

        b, a = signal.iirnotch(kwargs["f_notch"], kwargs["quality_factor"], sampling_rate)
        filtered_sig = signal.filtfilt(b, a, sig)
    else:
        raise ValueError(f'Missing keyword arguments for the selected method: "notch".')

    return filtered_sig


def _filter_ecg_pantompkins(sig: ArrayLike, sampling_rate: float) -> ArrayLike:
    """Filters ECG signal using the filter parameters defined in: Pan, J. & Tompkins, W. J.,(1985). 'A real-time QRS detection algorithm'."""

    W1 = 5 / (sampling_rate / 2)  # normalized frequency
    W2 = 15 / (sampling_rate / 2)  # normalized frequency
    N = 1
    btype = "bandpass"
    sos = signal.butter(N, [W1, W2], btype, output="sos")
    filtered_sig = signal.sosfiltfilt(sos, sig)

    return filtered_sig


def _filter_ecg_hamilton(sig: ArrayLike, sampling_rate: float) -> ArrayLike:
    """Filters ECG signal using the filter parameters defined in: Hamilton, P.S. (2002), 'Open Source ECG Analysis Software Documentation'."""

    W1 = 8 / (sampling_rate / 2)  # normalized frequency
    W2 = 16 / (sampling_rate / 2)  # normalized frequency
    N = 1
    btype = "bandpass"
    sos = signal.butter(N, [W1, W2], btype, output="sos")
    filtered_sig = signal.sosfiltfilt(sos, sig)

    return filtered_sig


def _filter_ecg_elgendi(sig: ArrayLike, sampling_rate: float) -> ArrayLike:
    """Filters ECG signal using the filter parameters defined in: Elgendi, M. & Jonkman, M. & De Boer, F. (2010). 'Frequency Bands Effects on QRS Detection'."""

    W1 = 8 / (sampling_rate / 2)  # normalized frequency
    W2 = 20 / (sampling_rate / 2)  # normalized frequency
    N = 2
    btype = "bandpass"
    sos = signal.butter(N, [W1, W2], btype, output="sos")
    filtered_sig = signal.sosfiltfilt(sos, sig)

    return filtered_sig
