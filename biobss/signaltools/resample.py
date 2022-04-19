import numpy as np
from scipy import signal
from numpy.typing import ArrayLike

def resample_signal(signal:ArrayLike,sample_rate:float,target_sample_rate:float) -> ArrayLike:
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
    
    resampled=signal.resample(signal,target_length)    
           
    return resampled