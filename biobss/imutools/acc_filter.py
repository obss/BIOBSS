from scipy import signal
from numpy.typing import ArrayLike
import warnings

def filter_acc(method):

    if method is None:      
        N = 2
        filter_type = 'lowpass'
        f_upper = 10
        warnings.warn(f"Default parameters will be used for filtering. {N}th order {method} {filter_type} filter with f2={f_upper}.")