from __future__ import annotations

import copy as copy

import numpy as np
from numpy.typing import ArrayLike


class Channel:
    """Biological signal channel class"""

    def __init__(self, signal: ArrayLike, name: str, sampling_rate: float):

        # Docstring
        """Biological signal channel class
        Parameters
        signal: ArrayLike
        name: str
            Name of the signal
        sampling_rate: float
            Sampling rate of the signal

        Attributes
        -----------
        channel: ArrayLike
            Signal
        signal_name: str
            Name of the signal
        sampling_rate: float
            Sampling rate of the signal
        """
        #

        # initialize channel data"""

        self.channel = np.array(signal)
        self.signal_name = name
        self.sampling_rate = sampling_rate

    def copy(self):
        # Docstring
        """Returns a copy of the channel
        Returns
        -------
        copy: Channel
            Copy of the channel
        """
        #
        return copy.deepcopy(self)

    def __eq__(self, other: object) -> bool:
        # Docstring
        """Check if two channels are equal
        Parameters
        ----------
        other: object
            Object to compare
        Returns
        -------
        equal: bool
            True if the two channels are equal, False otherwise
        """
        #
        if not isinstance(other, Channel):
            return False
        return (
            (self.signal_name == other.signal_name)
            and (np.array_equal(self.channel, other.channel))
            and self.sampling_rate == other.sampling_rate
        )

    def get_window(self, window_index):
        # Docstring
        """Returns a window of the channel
        Parameters
        ----------
        window_index: int
            Index of the window to return
        Returns
        -------
        window: ArrayLike
            Window of the channel
        """
        #
        if len(self.channel.shape) < 2:
            return self.channel
        else:
            return self.channel[window_index, :]

    def get_timestamp(self):
        if self.n_windows == 1:
            return np.array([0])
        else:
            return np.arange(self.n_windows)

    def get_window_timestamps(self):
        if self.n_windows == 1:
            return np.array([0])
        else:
            return np.arange(self.n_windows)

    @property
    def duration(self):
        if len(self.channel.shape) < 2:
            return self.channel.shape[0] / self.sampling_rate
        else:
            return self.channel.shape[1] / self.sampling_rate

    @property
    def n_windows(self):
        if len(self.channel.shape) < 2:
            return 1
        else:
            return self.channel.shape[0]

    @property
    def segmented(self):
        return self.n_windows > 1
