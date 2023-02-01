from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike


def segment_signal(
    signal: ArrayLike, sampling_rate: float, window_size: float, step_size=float, is_event=False
) -> ArrayLike:
    """Generates segments from input signal.

    Args:
        signal (ArrayLike): Signal to be segmented into windows.
        sampling_rate (float): Sampling rate of the signal.
        window_size (float): Size of signal windows in seconds.
        step_size (_type_, optional): Step Size in seconds.

    Raises:
        ValueError: If sampling rate is not greater than 0.
        ValueError: If signal is not an iterable.
        Exception:  If type of window size or step size is not int or float.
        Exception:  If window size or step size is not greater than 0.
        Exception:  If window size is greater than the length of input signal.

    Returns:
        ArrayLike: Collection of signal windows.
    """
    # Verify the inputs
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")
    if not isinstance(signal, Iterable):
        raise ValueError("Expecting an iterable")
    if not (
        (type(window_size) == type(0) or type(window_size) == type(0.0))
        and (type(step_size) == type(0) or type(step_size) == type(0.0))
    ):
        raise Exception("**ERROR** type(window_size) and type(step_size) must be int of float.")
    if window_size <= 0 or step_size <= 0:
        raise Exception("**ERROR** window_size and step_size must be positive.")
    if window_size * sampling_rate > len(signal):
        raise Exception(
            "**ERROR** window_size must be smaller than the length of signal. Make sure you entered window size in seconds."
        )
    # Number of windows
    window_size = int(window_size * sampling_rate)
    step_size = int(step_size * sampling_rate)
    num_frames = int(np.floor((len(signal) - window_size) / step_size) + 1)
    # Initialize the output signal
    signal_out = np.zeros((num_frames, window_size))
    # Sliding window operation
    for i in range(num_frames):
        signal_out[i] = signal[i * step_size : i * step_size + window_size]

    return signal_out
