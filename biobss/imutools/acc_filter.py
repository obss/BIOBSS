from scipy import signal
from numpy.typing import ArrayLike
import warnings

def filter_acc(sig: ArrayLike, sampling_rate: float, method: str='lowpass') -> ArrayLike:

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    method = method.lower()

    if method == 'lowpass':      
        N = 2
        btype = 'lowpass'
        W2 = 10 / (sampling_rate/2)
        warnings.warn(f"Default parameters will be used for filtering. {N}th order {method} {filter_type} filter with f2=10 Hz.")

        sos = signal.butter(N, W2, btype, output='sos')
        filtered_sig=signal.sosfiltfilt(sos, sig)

    else:
        raise ValueError(f"Undefined method: {method}.")

    return filtered_sig
