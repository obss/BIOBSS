import numpy as np
from scipy import signal as sg
from numpy.typing import ArrayLike
from ..pipeline.data_channel import Data_Channel

def resample_signal(signal:ArrayLike,sample_rate:float,target_sample_rate:float,return_time=False,t=None) -> ArrayLike:
    """_summary_

    Args:
        signal (1-D array): input signal
        sample_rate (float): Sample rate of input signal
        target_sample_rate (float): Expected sample rate after resampling

    Returns:
        1-D array: resampled signal
    """

    ratio=(target_sample_rate/sample_rate)
    target_length = round (len(signal) * ratio)
    
    if(return_time):
        if(t is None):
            t=np.arange(len(signal))/sample_rate
           
        resampled_x,resampled_t=sg.resample(signal,target_length,t=t)
        resampled=[resampled_x,resampled_t]
    else:
        resampled=sg.resample(signal,target_length)    
           
    return resampled


def resample_signal_object(signal:Data_Channel,target_sample_rate:float) -> Data_Channel:
    """_summary_

    Args:
        signal (Signal): input signal
        target_sample_rate (float): Expected sample rate after resampling

    Returns:
        Signal: resampled signal
    """
    if(not isinstance(signal,Data_Channel)):
        raise ValueError("Expecting a Signal object")
    
    signal.channel,signal.timestamp=resample_signal(signal.channel,signal.sampling_rate,target_sample_rate,return_time=True,t=signal.timestamp)
    signal.sampling_rate=target_sample_rate
    
    return signal
