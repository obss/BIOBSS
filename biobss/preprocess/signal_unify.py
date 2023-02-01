import numpy as np
from numpy.typing import ArrayLike


def unify_windows(
    windows: ArrayLike, timestamps: ArrayLike, window_size: float, step_size: float, sampling_rate: float
) -> tuple:
    """Unifies windows into a single array.

    Args:
        windows (ArrayLike): Collection of windows.
        timestamps (ArrayLike): Collection of timestamps.
        window_size (float): Window size.
        step_size (float): Step size.
        sampling_rate (float): Sampling rate of the signal (Hz).

    Raises:
        ValueError: If windows and timestamps do not have the same length.

    Returns:
        tuple: Unified signal, unified time array
    """
    window_size = int(window_size * sampling_rate)
    step_size = int(step_size * sampling_rate)

    if len(windows) != len(timestamps):
        raise ValueError("windows and timestamps must have the same length")

    num_windows = len(windows)
    num_points = num_windows * step_size + (window_size - step_size)

    unified_windows = np.zeros(num_points)
    unified_timestamps = np.zeros(num_points)
    unified_windows[:window_size] = windows[0]
    unified_timestamps[:window_size] = timestamps[0]

    for i in range(1, num_windows):
        unified_windows[i * step_size : i * step_size + window_size] = windows[i]
        unified_timestamps[i * step_size : i * step_size + window_size] = timestamps[i]

    return unified_windows, unified_timestamps
