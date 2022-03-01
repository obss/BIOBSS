import numpy as np
from scipy import signal



def resample_signal(signal,sample_rate,target_sample_rate):
    
    ratio=(target_sample_rate/sample_rate)
    target_length = round (len(signal) * ratio)
    
    resampled=signal.resample(signal,target_length)    
           
    return resampled