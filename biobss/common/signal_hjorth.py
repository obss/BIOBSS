from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike


def calculate_activity(sig: ArrayLike) -> float:
    """ Calculates Hjörth activity for the given signal.

    Args:
        sig (ArrayLike): Signal to be analyzed.

    Returns:
        float: Activity
    """

    return np.var(sig)


def _calculate_mobility(sig: ArrayLike) -> float:
    """Calculates Hjörth mobility for the given signal.

    Args:
        sig (ArrayLike): Signal to be analyzed.

    Returns:
        float: Mobility
    """
    f_derivative = np.gradient(sig, edge_order=1)
    mobility = np.square(np.var(f_derivative) / np.var(sig))
    return mobility


def calculate_complexity_mobility(sig: ArrayLike) -> Tuple:
    """ Calculates Hjörth complexity and mobility for the given signal.

    Mobility : The ratio of the variance of the first derivative of the signal to the variance of the signal.
    Complexity : The ratio of the variance of the second derivative of the signal to the variance of the signal.

    Args:
        sig (arraylike): Signal to be analyzed.

    Returns:
        Tuple: complexity, mobility
    """
    _mobility = _calculate_mobility(sig)
    f_derivative = np.gradient(sig, edge_order=1)
    complexity = _calculate_mobility(f_derivative) / _mobility
    return complexity, _mobility
