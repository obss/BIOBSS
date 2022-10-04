import numpy as np
from scipy import signal 
from numpy.typing import ArrayLike
from ..pipeline.data_channel import Data_Channel

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

def resample_signal_object(signal: Data_Channel, target_sample_rate: float) -> Data_Channel:
    """_summary_

    Args:
        signal (Signal): input signal
        target_sample_rate (float): Expected sample rate after resampling

    Returns:
        Signal: resampled signal
    """
    if(not isinstance(signal, Data_Channel)):
        raise ValueError("Expecting a Signal object")

    if(len(signal.channel.shape) < 2):
        signal.channel, signal.timestamp = resample_signal(
            signal.channel, signal.sampling_rate, target_sample_rate, return_time=True, t=signal.timestamp)
        signal.sampling_rate = target_sample_rate
    else:
        win_count = signal.channel.shape[0]
        target_length = round(
            signal.channel.shape[1] * (target_sample_rate/signal.sampling_rate))
        out = np.zeros((win_count, target_length))
        out_ts = np.zeros((win_count, target_length))
        for w in range(signal.channel.shape[0]):
            out[w], out_ts[w] = resample_signal(
                signal.channel[w], signal.sampling_rate, target_sample_rate, return_time=True, t=signal.timestamp[w])
        signal.channel = out
        signal.timestamp = out_ts
        signal.sampling_rate = target_sample_rate

    return signal
