from scipy import signal
from numpy.typing import ArrayLike
import warnings

def filter_signal(sig: ArrayLike, sampling_rate: float, filter_type: str, N: int, f_lower: float=None, f_upper: float=None, axis: int=0, signal_type: str=None, method: str=None) -> ArrayLike:
    """Filters a signal using a N-th order Butterworth filter unless signal_type is specified. If signal_type is specified, predefined filter parameters are used.

    Args:
        sig (ArrayLike): Signal to be filtered.
        sampling_rate (float): The sampling frequency of the signal (Hz).
        filter_type (str): Type of the filter. Can be 'lowpass', 'highpass' or 'bandpass'.
        N (int): Order of the filter. 
        f_lower (float, optional): Lower cutoff frequency (Hz). Defaults to None.
        f_upper (float, optional): Upper cutoff frequency (Hz). Defaults to None.
        axis (int, optional): The axis alongh which filtering is applied. Defaults to 0.
        signal_type (str, optional): Type of the input signal. If None, a Butterworth filter is used and the filter parameters should be defined. If a value is passed, the default filter parameters for each signal type (most common in the literature) are used for filtering. Defaults to None.
        method (str, optional): Filtering method for the selected signal_type. Defaults to None.

    Raises:
        ValueError: If sampling rate is less than or equal to zero.
        ValueError: If upper cutoff frequency is not provided when filter_type is 'lowpass'.
        ValueError: If lower cutoff frequency is not provided when filter_type is 'highpass'.
        ValueError: If lower or upper cut-off frequency is less than zero.
        ValueError: If lower and upper cutoff frequencies are not provided when filter_type is 'bandpass'.
        ValueError: If filter_type is not given as one of 'lowpass', 'highpass' or 'bandpass'.
        ValueError: If signal type is not one of valid types.

    Returns:
        ArrayLike: Filtered signal.
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    valid_types = ['ECG','ACC','EDA','PPG']

    if signal_type is None:

        if filter_type=='lowpass':
            if f_upper is not None:
                if f_upper < 0:
                    raise ValueError("Cut-off frequency must be greater than 0.")
                W2=f_upper/(sampling_rate/2) #normalized frequency
                btype='lowpass'
                sos2 = signal.butter(N, W2, btype, output='sos')
                filtered_sig=signal.sosfiltfilt(sos2, sig, axis=axis)
            else:
                raise ValueError("Upper cutoff frequency is required for lowpass filtering.")

        elif filter_type=='highpass':
            if f_lower is not None:
                if f_lower < 0:
                    raise ValueError("Cut-off frequency must be greater than 0.")
                W1=f_lower/(sampling_rate/2) #normalized frequency
                btype='highpass'
                sos1 = signal.butter(N, W1, btype, output='sos')
                filtered_sig=signal.sosfiltfilt(sos1, sig, axis=axis)
            else:
                raise ValueError("Lower cutoff frequency is required for highpass filtering.")

        elif filter_type=='bandpass':
            if f_lower is not None and f_upper is not None:
                if f_lower < 0 or f_upper < 0:
                    raise ValueError("Cut-off frequencies must be greater than 0.")

                W1=f_lower/(sampling_rate/2) #normalized frequency 
                W2=f_upper/(sampling_rate/2) #normalized frequency
                btype='bandpass'
                sos = signal.butter(N, [W1,W2], btype, output='sos')
                filtered_sig=signal.sosfiltfilt(sos, sig, axis=axis)
            else:
                raise ValueError("Both lower and upper cutoff frequencies are required for bandpass filtering.")

        else:
            raise ValueError("Filter type should be one of 'lowpass', 'highpass' or 'bandpass'.")

    else:
        signal_type = signal_type.upper()
        if method is not None:
            method = method.lower()
        
        if signal_type == 'ECG':
            filtered_sig = _filter_ecg(method=method) 

        elif signal_type == 'PPG':
            filtered_sig = _filter_ppg(method=method)

        elif signal_type == 'ACC':
            filtered_sig = _filter_acc(method=method)

        elif signal_type == 'EDA':
            filtered_sig = _filter_eda(method=method)

        else:
            raise ValueError(f"Signal type should be one of {valid_types}.")

    return filtered_sig


def _filter_ppg(method):

    if method is None:
        N = 2
        filter_type = 'bandpass'
        f_lower = 0.5
        f_upper = 5
        warnings.warn(f"Default parameters will be used for filtering. {N}th order {method} {filter_type} filter with f1={f_lower} and f2={f_upper}.")

def _filter_eda(method):

    if method is None:
        N = 2
        filter_type = 'bandpass'
        f_lower = 0.5
        f_upper = 5
        warnings.warn(f"Default parameters will be used for filtering. {N}th order {method} {filter_type} filter with f1={f_lower} and f2={f_upper}.")

def _filter_ecg(method):

    if method is None:  
        N = 2
        filter_type = 'bandpass'
        f_lower = 0.5
        f_upper = 5
        warnings.warn(f"Default parameters will be used for filtering. {N}th order {method} {filter_type} filter with f1={f_lower} and f2={f_upper}.")


def _filter_acc(method):

    if method is None:      
        N = 2
        filter_type = 'lowpass'
        f_upper = 10
        warnings.warn(f"Default parameters will be used for filtering. {N}th order {method} {filter_type} filter with f2={f_upper}.")