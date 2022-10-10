from scipy import signal
from numpy.typing import ArrayLike
import warnings

def filter_ppg(method):

    if method is None:
        N = 2
        filter_type = 'bandpass'
        f_lower = 0.5
        f_upper = 5
        warnings.warn(f"Default parameters will be used for filtering. {N}th order {method} {filter_type} filter with f1={f_lower} and f2={f_upper}.")