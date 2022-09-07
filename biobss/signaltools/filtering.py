from scipy import signal
from numpy.typing import ArrayLike

def filter_signal(sig: ArrayLike, filter_type: str, N: int, sampling_rate: float, f_lower: float=None, f_upper: float=None, axis: int=0) -> ArrayLike:
    """Filters a signal using a N-th order Butterworth filter

    Args:
        sig (ArrayLike): Signal to be filtered.
        filter_type (str): Type of the filter. Can be 'lowpass', 'highpass' or 'bandpass'.
        N (int): Order of the filter. 
        sampling_rate (float): The sampling frequency of the signal (Hz).
        f_lower (float, optional): Lower cutoff frequency (Hz). Defaults to None.
        f_upper (float, optional): Upper cutoff frequency (Hz). Defaults to None.
        axis (int, optional): The axis alongh which filtering is applied. Defaults to 0.

    Raises:
        ValueError: If upper cutoff frequency is not provided when filter_type is 'lowpass'.
        ValueError: If lower cutoff frequency is not provided when filter_type is 'highpass'.
        ValueError: If lower and upper cutoff frequencies are not provided when filter_type is 'bandpass'.
        ValueError: If filter_type is not given as one of 'lowpass', 'highpass' or 'bandpass'.

    Returns:
        ArrayLike: Filtered signal.
    """
 
    if filter_type=='lowpass':
        if f_upper is not None:
            W2=f_upper/(sampling_rate/2) #normalized frequency
            btype='lowpass'
            b2,a2 = signal.butter(N,W2,btype)
            filtered_sig=signal.filtfilt(b2,a2,sig,axis=axis)
        else:
            raise ValueError("Upper cutoff frequency is required for lowpass filtering.")

    elif filter_type=='highpass':
        if f_lower is not None:
            W1=f_lower/(sampling_rate/2) #normalized frequency
            btype='highpass'
            b1,a1 = signal.butter(N,W1,btype)
            filtered_sig=signal.filtfilt(b1,a1,sig,axis=axis)
        else:
            raise ValueError("Lower cutoff frequency is required for highpass filtering.")

    elif filter_type=='bandpass':
        if f_lower is not None and f_upper is not None: 
            W2=f_upper/(sampling_rate/2) #normalized frequency
            btype='lowpass'
            b2,a2 = signal.butter(N,W2,btype)
            temp_sig=signal.filtfilt(b2,a2,sig,axis=axis)       

            W1=f_lower/(sampling_rate/2) #normalized frequency
            btype='highpass'
            b1,a1 = signal.butter(N,W1,btype)
            filtered_sig=signal.filtfilt(b1,a1,temp_sig,axis=axis)
        else:
            raise ValueError("Both lower and upper cutoff frequencies are required for bandpass filtering.")

    else:
        raise ValueError("Filter type should be one of 'lowpass', 'highpass' or 'bandpass'.")

    return filtered_sig


    