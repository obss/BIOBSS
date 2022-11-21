from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike


def calculate_activity(signal: ArrayLike) -> float:
    """ This method returns Hjörth activity for the given signal.

    Args:
        signal (ArrayLike): input signal

    Returns:
        float: Activity
    """

    return np.var(signal)


def calculate_mobility(signal: ArrayLike) -> float:
    """This method returns Hjörth mobility for the given signal.

    Args:
        signal (ArrayLike): input signal

    Returns:
        float: Mobility
    """
    f_derivative = np.gradient(signal, edge_order=1)
    mobility = np.square(np.var(f_derivative) / np.var(signal))
    return mobility


def calculate_complexity(signal: ArrayLike) -> Tuple:
    """ This method returns Hjörth complexity and mobility for the given signal.

    Mobility : The ratio of the variance of the first derivative of the signal to the variance of the signal.
    Complexity : The ratio of the variance of the second derivative of the signal to the variance of the signal.

    Args:
        signal (arraylike): input signal

    Returns:
        Tuple: complexity, mobility
    """
    _mobility = calculate_mobility(signal)
    f_derivative = np.gradient(signal, edge_order=1)
    complexity = calculate_mobility(f_derivative) / _mobility
    return complexity, _mobility
