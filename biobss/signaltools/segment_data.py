import numpy as np

def naive_segment(signal,window_size,overlap_size=1,sampling_rate=20):
    """[summary]

    Args:
        signal ([type]): [Any signal to be segmented into windows]
        window_size ([type]): [Size of signal windows in seconds]
        overlap_ratio (float, optional): [description]. Defaults to 0.5.
        sampling_rate (int, optional): [Frequency of the signal]. Defaults to 20.

    Returns:
        [type]: [Collection of signal windows]
    """       
    step=overlap_size*sampling_rate
    
    window_length=window_size*sampling_rate
    segmented=[]
    for i in range(0,len(signal)-window_length,step):
        segmented.append(signal[i:i+window_length])
    return np.array(segmented)