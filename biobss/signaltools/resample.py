import numpy as np
from scipy import signal 
from numpy.typing import ArrayLike


def resample_signal(signal: ArrayLike, sampling_rate: float, target_sampling_rate: float, return_time: bool=False, t: ArrayLike=None) -> ArrayLike:
    """Resamples the given signal.

    Args:
        signal (ArrayLike): Signal to be analyzed.
        sample_rate (float): Sampling rate of the signal.
        target_sample_rate (float): Expected sample rate after resampling.
        return_time (bool, optional): If True, time array is returned. Defaults to False.
        t (ArrayLike, optional): Time array. Defaults to None.

    Returns:
        ArrayLike: Resampled signal.
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")
        
    if target_sampling_rate <= 0:
        raise ValueError("Target sampling rate must be greater than 0.")        

    signal = np.array(signal)
    ratio = (target_sampling_rate/sampling_rate)
    target_length = round(len(signal) * ratio)

    if return_time:
        if(t is None):
            t = np.arange(len(signal))/sampling_rate

        resampled_x, resampled_t = signal.resample(signal, target_length, t=t)
        resampled = [resampled_x, resampled_t]
    else:
        resampled = signal.resample(signal, target_length)

    return resampled


