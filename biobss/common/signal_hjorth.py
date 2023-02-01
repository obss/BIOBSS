from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike


def hjorth_activity(sig: ArrayLike) -> float:
    """Calculates Hjörth activity for the given signal.
    Activity parameter represents the signal power.

    Args:
        sig (ArrayLike): Signal to be analyzed.

    Returns:
        float: Activity
    """

    return np.var(sig)


def hjorth_complexity_mobility(sig: ArrayLike) -> Tuple:
    """Calculates Hjörth complexity and mobility for the given signal.
    Mobility represents the mean frequency and complexity represents the change in frequency.

    Args:
        sig (arraylike): Signal to be analyzed.

    Returns:
        Tuple: complexity, mobility
    """

    _mobility = _hjorth_mobility(sig)
    f_derivative = np.gradient(sig, edge_order=1)
    complexity = _hjorth_mobility(f_derivative) / _mobility

    return complexity, _mobility


def _hjorth_mobility(sig: ArrayLike) -> float:
    """Calculates Hjorth mobility."""
    f_derivative = np.gradient(sig, edge_order=1)
    mobility = np.sqrt(np.var(f_derivative) / np.var(sig))

    return mobility
