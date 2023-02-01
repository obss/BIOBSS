from __future__ import annotations

import copy as copy

import numpy as np
from numpy.typing import ArrayLike


class Event_Channel:
    """Biological signal channel class"""

    def __init__(self, events: list, name: str, sampling_rate: float):

        if not isinstance(events, list):
            raise ValueError("events data must be a list")
        elif len(events) == 0:
            raise ValueError("events must have at least one key")
        elif not isinstance(name, str):
            raise ValueError("name must be a string")
        elif not isinstance(sampling_rate, (float, int)):
            raise ValueError("sampling_rate must be a float or int")

        self.channel = events
        self.signal_name = name
        self.sampling_rate = sampling_rate

    def get_event(self, event_name: str):
        if event_name in self.channel.keys():
            return self.channel[event_name]
        else:
            raise ValueError("event_name not found in channel")

    def get_window(self, window_index):

        if self.n_windows == 1:
            return self.channel
        else:
            return self.channel[window_index]

    def __getitem__(self, event_name: str):
        return self.get_event(event_name)

    def __setitem__(self, event: ArrayLike):
        if not isinstance(event, (np.ndarray, list)):
            raise ValueError("event must be a list or numpy array")
        else:
            self.channel = list(event)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Event_Channel):
            raise ValueError("other must be an Event_Channel object")
        return (
            self.channel == other.channel
            and self.signal_name == other.signal_name
            and self.sampling_rate == other.sampling_rate
        )

    def copy(self):
        return copy.deepcopy(self)

    def __copy__(self):
        return self.copy()

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
    def n_events(self):
        return len(self.channel.keys())

    @property
    def event_names(self):
        return list(self.channel.keys())

    @property
    def n_windows(self):
        if all(isinstance(value, (np.ndarray, list)) for value in self.channel):
            return len(self.channel)
        else:
            return 1

    @property
    def segmented(self):
        return self.n_windows > 1
