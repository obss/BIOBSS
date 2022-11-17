from numpy.typing import ArrayLike
from scipy import signal

def filter_ecg(sig: ArrayLike, sampling_rate: float, method: str, **kwargs) -> ArrayLike:
  """Filters ECG signal using predefined filters.

  Args:
      sig (ArrayLike): ECG signal.
      sampling_rate (float): Sampling rate of the ECG signal.
      method (str): Filtering method. Should be one of ['notch', 'bandpass', 'pantompkins', 'hamilton', 'elgendi].

  Raises:
      ValueError: If sampling rate is less than or equal to 0. 
      ValueError: If cut-off frequency is less than 0.
      ValueError: If required parameters are not provided for the selected method.
      ValueError: If filtering method is not one of ['notch', 'pantompkins', 'hamilton', 'elgendi].

  Returns:
      ArrayLike: Filtered ECG signal.
  """

  if sampling_rate <= 0:
    raise ValueError("Sampling rate must be greater than 0.")

  method = method.lower()

  if method == 'notch':
    if all(k in kwargs.keys() for k in ('f_notch', 'quality_factor')):
      if kwargs['f_notch'] <= 0:
        raise ValueError("Cut-off frequencies must be greater than 0.")

      b, a = signal.iirnotch(kwargs['f_notch'], kwargs['quality_factor'], sampling_rate)
      filtered_sig = signal.filtfilt(b, a, sig)
    else:
      raise ValueError(f'Missing keyword arguments for method: {method}.')

  elif method == 'pantompkins':

    W1=5/(sampling_rate/2) #normalized frequency 
    W2=15/(sampling_rate/2) #normalized frequency
    N = 1
    btype = 'bandpass'
    sos = signal.butter(N, [W1,W2], btype, output='sos')
    filtered_sig=signal.sosfiltfilt(sos, sig)

  elif method =='hamilton':

    W1=8/(sampling_rate/2) #normalized frequency 
    W2=16/(sampling_rate/2) #normalized frequency
    N = 1 
    btype = 'bandpass' 
    sos = signal.butter(N, [W1,W2], btype, output='sos')
    filtered_sig=signal.sosfiltfilt(sos, sig)

  elif method == 'elgendi':

    W1=8/(sampling_rate/2) #normalized frequency 
    W2=20/(sampling_rate/2) #normalized frequency
    N = 2
    btype = 'bandpass'
    sos = signal.butter(N, [W1,W2], btype, output='sos')
    filtered_sig=signal.sosfiltfilt(sos, sig)

  else:
    raise ValueError(f"Undefined method: {method}.")

  return filtered_sig