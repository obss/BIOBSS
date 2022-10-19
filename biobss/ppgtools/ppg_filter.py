from scipy import signal
from numpy.typing import ArrayLike
import warnings

def filter_ppg(sig: ArrayLike, sampling_rate: float, method: str='bandpass') -> ArrayLike:
    
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    method = method.lower()

    if method == 'bandpass':
        N = 2
        btype = 'bandpass'
        W1 = 0.5 / (sampling_rate/2)
        W2 = 5 / (sampling_rate/2)
        warnings.warn(f"Default parameters will be used for filtering. {N}th order {method} {btype} filter with f1=0.5 Hz and f2=5 Hz.")

        sos = signal.butter(N, [W1,W2], btype, output='sos')
        filtered_sig=signal.sosfiltfilt(sos, sig)

    else:
        raise ValueError(f"Undefined method: {method}.")

    return filtered_sig
