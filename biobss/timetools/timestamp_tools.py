import numpy as np
from numpy.typing import ArrayLike


def create_timestamp_signal(resolution: str, length: float, start: float, rate: float) -> ArrayLike:
    """Generates a timestamp array.

    Args:
        resolution (str): Timestamp resolution. It can be 'ns', 'ms', 's' or 'min'.
        length (float): Length of timestamp array to be generated.
        start (float): Starting time.
        rate (float): Rate of increment.

    Raises:
        ValueError: If starting time is less then zero.
        ValueError: If resolution is undefined.

    Returns:
        ArrayLike: Timestamp array.
    """

    if start < 0:
        raise ValueError("Timestamp start must be greater than 0")

    if resolution == "ns":
        timestamp_factor = 1 / 1e-9
    elif resolution == "ms":
        timestamp_factor = 1 / 0.001
    elif resolution == "s":
        timestamp_factor = 1
    elif resolution == "min":
        timestamp_factor = 60
    else:
        raise ValueError('resolution must be "ns","ms","s","min"')

    timestamp = (np.arange(length) / rate) * timestamp_factor
    timestamp = timestamp + start

    return timestamp


def check_timestamp(timestamp, timestamp_resolution):

    possible_timestamp_resolution = ["ns", "ms", "s", "min"]

    if timestamp_resolution in possible_timestamp_resolution:
        pass
    else:
        raise ValueError('timestamp_resolution must be "ns","ms","s","min"')

    if np.any(np.diff(timestamp) < 0):
        raise ValueError("Timestamp must be monotonic")

    return True
