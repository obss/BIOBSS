import numpy as np
from numpy.typing import ArrayLike
from collections.abc import Iterable

def segment_signal(signal: ArrayLike, window_size: float, step_size=1., sampling_rate=20.) -> ArrayLike:
    """Generates segments from input signal.

    Args:
        signal (1-D arraylike): Signal to be segmented into windows.
        window_size (float): Size of signal windows in seconds.
        step_size (float, optional): Step Size in seconds. Defaults to 1.
        sampling_rate (float, optional): Sampling rate of the signal. Defaults to 20.

    Returns:
        [2-D array]: [Collection of signal windows]
    """
    # Verify the inputs
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")
    if not isinstance(signal, Iterable):
        raise ValueError("Expecting an iterable")
    if not ((type(window_size) == type(0) or type(window_size) == type(0.)) and (type(step_size) == type(0) or type(step_size) == type(0.))):
        raise Exception(
            "**ERROR** type(window_size) and type(step_size) must be int of float.")
    if window_size <= 0 or step_size <= 0:
        raise Exception(
            "**ERROR** window_size and step_size must be positive.")
    if window_size*sampling_rate > len(signal):
        raise Exception(
            "**ERROR** window_size must be smaller than the length of signal. Make sure you entered window size in seconds.")
    # Number of windows
    window_size = int(window_size*sampling_rate)
    step_size = int(step_size*sampling_rate)
    num_frames = int(np.floor((len(signal) - window_size) / step_size) + 1)
    # Initialize the output signal
    signal_out = np.zeros((num_frames, window_size))
    # Sliding window operation
    for i in range(num_frames):
        signal_out[i] = signal[i * step_size:i * step_size + window_size]
    return signal_out
