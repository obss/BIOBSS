import numpy as np

def naive_segment(signal,window_size,step_size=1,sampling_rate=20):
    """[summary]

    Args:
        signal ([1-D arraylike]): [Any signal to be segmented into windows]
        window_size ([type]): [Size of signal windows in seconds]
        step_size (float, optional): [Step Size in seconds]. Defaults to 1.
        sampling_rate (int, optional): [Frequency of the signal]. Defaults to 20.

    Returns:
        [2-D array]: [Collection of signal windows]
    """       
    step=step_size*sampling_rate
    
    window_length=window_size*sampling_rate
    segmented=[]
    for i in range(0,len(signal)-window_length,step):
        segmented.append(signal[i:i+window_length])
    return np.array(segmented)