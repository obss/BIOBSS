import numpy as np
from scipy import signal



def resample_signal(signal,sample_rate,target_sample_rate):
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