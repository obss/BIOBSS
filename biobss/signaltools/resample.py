import numpy as np
from scipy import signal as sg
from numpy.typing import ArrayLike
from ..pipeline.data_channel import Data_Channel


def resample_signal(signal: ArrayLike, sample_rate: float, target_sample_rate: float, return_time=False, t=None) -> ArrayLike:
    """_summary_

    Args:
        signal (1-D array): input signal
        sample_rate (float): Sample rate of input signal
        target_sample_rate (float): Expected sample rate after resampling

    Returns:
        1-D array: resampled signal
    """
    signal = np.array(signal)
    ratio = (target_sample_rate/sample_rate)
    target_length = round(len(signal) * ratio)

    if return_time:
        if(t is None):
            t = np.arange(len(signal))/sample_rate

        resampled_x, resampled_t = sg.resample(signal, target_length, t=t)
        resampled = [resampled_x, resampled_t]
    else:
        resampled = sg.resample(signal, target_length)

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
