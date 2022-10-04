from numpy.typing import ArrayLike
from ecgdetectors import Detectors
import neurokit2 as nk
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
      ValueError: If filtering method is not one of ['notch', 'bandpass', 'pantompkins', 'hamilton', 'elgendi].

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

  elif method == 'bandpass':
    if all(k in kwargs.keys() for k in ('f_lower', 'f_upper', 'N')):
      if kwargs['f_lower'] < 0 or kwargs['f_upper'] < 0:
          raise ValueError("Cut-off frequencies must be greater than 0.")

      W1=kwargs['f_lower']/(sampling_rate/2) #normalized frequency 
      W2=kwargs['f_upper']/(sampling_rate/2) #normalized frequency
      btype='bandpass'
      sos = signal.butter(kwargs['N'], [W1,W2], btype, output='sos')
      filtered_sig=signal.sosfiltfilt(sos, sig)
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
 

def ecg_peaks(sig: ArrayLike , sampling_rate: float, method: str='pantompkins') -> ArrayLike:
  """Detects R peaks from ECG signal. 
     Uses py-ecg-detectors package(https://github.com/berndporr/py-ecg-detectors/).
  Args:
      sig (ArrayLike): Unfiltered ECG signal.
      sampling_rate (float): Sampling rate of the ECG signal.
      method (str, optional): Peak detection method. Should be 'pantompkins', 'hamilton' or 'elgendi'. Defaults to 'pantompkins'.
        'pantompkins': "Pan, J. & Tompkins, W. J.,(1985). 'A real-time QRS detection algorithm'. IEEE transactions
                        on biomedical engineering, (3), 230-236."
        'hamilton': "Hamilton, P.S. (2002), 'Open Source ECG Analysis Software Documentation', E.P.Limited."
        'elgendi': "Elgendi, M. & Jonkman, M. & De Boer, F. (2010). 'Frequency Bands Effects on QRS Detection',
                    The 3rd International Conference on Bio-inspired Systems and Signal Processing (BIOSIGNALS2010). 428-431.

  Raises:
      ValueError: If sampling rate is not greater than 0.
      ValueError: If method is not 'pantompkins', 'hamilton' or 'elgendi'.

  Returns:
      ArrayLike: R-peak locations
  """

  if sampling_rate <= 0:
    raise ValueError("Sampling rate must be greater than 0.")

  method = method.lower()
  detectors = Detectors(sampling_rate)

  if method == 'pantompkins':
    r_peaks = detectors.pan_tompkins_detector(sig)

  elif method == 'hamilton':
    r_peaks = detectors.hamilton_detector(sig)

  elif method == 'elgendi':
    r_peaks = detectors.two_average_detector(sig)

  else:
    raise ValueError(f"Undefined method: {method}")

  return r_peaks


def ecg_waves(sig: ArrayLike, peaks_locs, sampling_rate: float, delineator: str='neurokit2') -> dict:
  """Detects fiducial points of ECG signal and returns a dictionary.

  Args:
      sig (ArrayLike): ECG signal.
      peaks_locs (_type_): R peak locations.
      sampling_rate (float): Sampling rate of the ECG signal.
      delineator (str, optional): Delineator to be used. Defaults to 'neurokit2'.

  Raises:
      ValueError: If sampling rate is less than or equal to 0. 
      ValueError: If delineator is not one of defined methods.

  Returns:
      dict: _description_
  """
  if sampling_rate <= 0:
    raise ValueError("Sampling rate must be greater than 0.")

  delineator = delineator.lower()

  if delineator == "neurokit2":
    info = nk.ecg_delineate(sig, rpeaks=peaks_locs, sampling_rate=sampling_rate, method="peak")

  else:
    raise ValueError(f"Undefined delineator {delineator}.")

  return info