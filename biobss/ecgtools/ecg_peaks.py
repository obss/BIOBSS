import neurokit2 as nk
from ecgdetectors import Detectors
from numpy.typing import ArrayLike


def ecg_detectpeaks(sig: ArrayLike, sampling_rate: float, method: str = "pantompkins") -> ArrayLike:
    """Detects R peaks from ECG signal.
    Uses py-ecg-detectors package(https://github.com/berndporr/py-ecg-detectors/).

    Args:
        sig (ArrayLike): ECG signal.
        sampling_rate (float): Sampling rate of the ECG signal (Hz).
        method (str, optional): Peak detection method. Should be 'pantompkins', 'hamilton' or 'elgendi'. Defaults to 'pantompkins'.
        'pantompkins': "Pan, J. & Tompkins, W. J.,(1985). 'A real-time QRS detection algorithm'. IEEE transactions
                        on biomedical engineering, (3), 230-236."
        'hamilton': "Hamilton, P.S. (2002), 'Open Source ECG Analysis Software Documentation', E.P.Limited."
        'elgendi': "Elgendi, M. & Jonkman, M. & De Boer, F. (2010). 'Frequency Bands Effects on QRS Detection',
                    The 3rd International Conference on Bio-inspired Systems and Signal Processing (BIOSIGNALS2010). 428-431.

    Raises:
        ValueError: If sampling rate is not greater than 0.
        ValueError: If method is not 'pantompkins', 'hamilton' or 'elgendi'.

    Returns:
        ArrayLike: R-peak locations
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    method = method.lower()
    detectors = Detectors(sampling_rate)

    if method == "pantompkins":
        r_peaks = detectors.pan_tompkins_detector(sig)
    elif method == "hamilton":
        r_peaks = detectors.hamilton_detector(sig)
    elif method == "elgendi":
        r_peaks = detectors.two_average_detector(sig)
    else:
        raise ValueError(f"Undefined method: {method}")

    return r_peaks
