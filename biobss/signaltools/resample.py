import numpy as np
from scipy import signal
from numpy.typing import ArrayLike
from ..pipeline.signal import Signal

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
           
        resampled_x,resampled_t=signal.resample(signal,target_length,t=t)
        resampled=[resampled_x,resampled_t]
    else:
        resampled=signal.resample(signal,target_length)    
           
    return resampled


def resample_signal_object(signal:Signal,target_sample_rate:float) -> Signal:
    """_summary_

    Args:
        signal (Signal): input signal
        target_sample_rate (float): Expected sample rate after resampling

    Returns:
        Signal: resampled signal
    """
    
    resampled=resample_signal(signal.signal,signal.sampling_rate,target_sample_rate)
    resampled=Signal(resampled,target_sample_rate,signal.modality,signal.signal_name)
    
    return resampled
