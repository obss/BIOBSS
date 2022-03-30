from scipy import signal
from numpy.typing import ArrayLike

def filter_signal(sig: ArrayLike, filter_type: str, N: int, fs: float, f1: float=None,f2: float=None, axis=0) -> dict:
    """Filters the signal using a Butterworth filter

    Args:
        sig (array): Signal to be filtered
        filter_type (str): Low-pass, high-pass or band-pass 
        N (int): Filter order
        fs (float): Sampling rate
        f1 (float, optional): Lower cutoff frequency. Defaults to None.
        f2 (float, optional): Higher cutoff frequency. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        (array): filtered signal
    """

    if filter_type=='lowpass':

        if f2 is not None:

            W2=f2/(fs/2) #normalized frequency
            btype='lowpass'
            b2,a2 = signal.butter(N,W2,btype)
            filtered_sig=signal.filtfilt(b2,a2,sig,axis=axis)
        else:
            raise ValueError("f2 is required.")

    elif filter_type=='highpass':

        if f1 is not None:
            W1=f1/(fs/2) #normalized frequency
            btype='highpass'
            b1,a1 = signal.butter(N,W1,btype)
            filtered_sig=signal.filtfilt(b1,a1,sig,axis=axis)
        else:
            raise ValueError("f1 is required.")

    elif filter_type=='bandpass':

        if f1 is not None and f2 is not None: 
            W2=f2/(fs/2) #normalized frequency
            btype='lowpass'
            b2,a2 = signal.butter(N,W2,btype)
            temp_sig=signal.filtfilt(b2,a2,sig,axis=axis)       

            W1=f1/(fs/2) #normalized frequency
            btype='highpass'
            b1,a1 = signal.butter(N,W1,btype)
            filtered_sig=signal.filtfilt(b1,a1,temp_sig,axis=axis)
        else:
            raise ValueError("f1 and f2 are required.")

    else:

        raise ValueError("Filter type error.")



    return filtered_sig


    