from numpy.typing import ArrayLike

def normalize_signal(signal : ArrayLike,method='zscore') -> ArrayLike:
    """Method for normalizing given signal

    Args:
        signal (arraylike): _description_
        method (str, optional): Normalization method. Defaults to 'zscore'.

    Returns:
        1-D array: normalized signal
    """
    
    # Need to add signal check
    if(method=='zscore'):
        return (signal-signal.mean())/signal.std()
    elif(method=='minmax'):
        return (signal-signal.min())/(signal.max()-signal.min())
    else:
        print("Normalization method error! Using Default (Zscore)")
        return (signal-signal.mean())/signal.std()