import copy as copy
import warnings
from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .bio_channel import Channel
from .event_channel import Event_Channel


class Bio_Data:
    """Signal object with add and iterate process objects"""

    def __init__(self):
        # Docstring
        """Signal object with add and iterate process objects
        Attributes
        -----------
        channels: dict
            Dictionary of channels
        """
        #
        self.channels = {}  # Data_Channel objects

    def add_channel(
        self,
        signal: Union[ArrayLike, Channel, Event_Channel],
        channel_name: str = None,
        sampling_rate: Union[int, float] = None,
        modify_existed: bool = False,
        is_event=False,
    ):
        # Docstrin
        """Add a channel to the signal
        Parameters
        ----------
        signal: Union[ArrayLike, Channel ,Event_Channel]
            Signal to add
        channel_name: str
            Name of the channel
        sampling_rate: Union[int, float]
            Sampling rate of the signal
        modify_existed: bool
            If True, if a channel with the same name already exists, it will be overwritten
            If False, if a channel with the same name already exists, a new name will be generated
        is_event: bool
            If True, the signal will be converted to an Event_Channel object
            If False, the signal will be converted to a Channel object
        """
        #
        if isinstance(signal, pd.DataFrame):
            if channel_name is None:
                channel_name = "channel"
            if channel_name in self.channels.keys():
                if modify_existed:
                    warnings.warn("Overwriting channel " + channel_name)
                else:
                    i = 1
                    while channel_name in self.channels.keys():
                        channel_name = channel_name + "_" + str(i)
                        i = i + 1
            self.channels[channel_name] = Event_Channel(signal, channel_name, sampling_rate)
            return

        if isinstance(signal, (Channel, Event_Channel)):
            if channel_name is None:
                channel_name = signal.signal_name
            if channel_name in self.channels.keys():
                if modify_existed:
                    warnings.warn("Overwriting channel " + channel_name)
                else:
                    i = 1
                    while channel_name in self.channels.keys():
                        channel_name = channel_name + "_" + str(i)
                        i = i + 1
            self.channels[channel_name] = signal.copy()

        else:
            if isinstance(signal, (np.ndarray, list)):
                signal = np.array(signal)
            if sampling_rate is None:
                raise ValueError("sampling_rate must be provided if signal is not a Channel or Event_Channel object")
            if not isinstance(sampling_rate, (int, float)):
                raise ValueError("sampling_rate must be a float or integer")
            if channel_name is None:
                channel_name = "channel"
            if channel_name in self.channels.keys():
                if modify_existed:
                    warnings.warn("Overwriting channel " + channel_name)
                else:
                    i = 1
                    while channel_name in self.channels.keys():
                        channel_name = channel_name + "_" + str(i)
                        i = i + 1
            if is_event:
                self.channels[channel_name] = Event_Channel(signal, channel_name, sampling_rate)
            else:
                self.channels[channel_name] = Channel(signal, channel_name, sampling_rate)

    def remove_channel(self, channel_name):
        # Docstring
        """Remove a channel from the signal
        Parameters
        -----------
        channel_name: str
            Name of the channel to remove
        """
        #
        if channel_name not in self.channels.keys():
            raise ValueError("Channel does not exist")
        self.channels.pop(channel_name)

    def rename_channel(self, old_name, new_name):
        # Docstring
        """Rename a channel
        Parameters
        -----------
        old_name: str
            Old name of the channel
        new_name: str
            New name of the channel
        """
        #
        if old_name not in self.channels.keys():
            raise ValueError("Channel does not exist")
        if new_name in self.channels.keys():
            raise ValueError("Channel name already exists")
        self.channels[new_name] = self.channels.pop(old_name)

    def get_channel_names(self):
        return list(self.channels.keys())

    def copy(self):
        return copy.deepcopy(self)

    def join(self, other: "Bio_Data", overwrite: bool = False):
        # Docstring
        """Join two Bio_Data objects
        Parameters
        -----------
        other: Bio_Data
            Bio_Data object to join
        overwrite: bool
            If True, if a channel with the same name already exists, it will be overwritten
            If False, if a channel with the same name already exists, a new name will be generated
        """
        #
        if not isinstance(other, Bio_Data):
            raise ValueError("Can only join Bio_Data objects")
        other = other.copy()
        for k, v in other.channels.items():
            if k in self.channels.keys():
                if overwrite:
                    warnings.warn("Overwriting channel " + k)
                else:
                    k = k + "_1"
            self.channels[k] = v.copy()

        return self

    def __getitem__(self, key: Union[str, int]) -> Union[Channel, Event_Channel]:
        if isinstance(key, str):
            return self.channels[key]
        elif isinstance(key, int):
            return self.channels[list(self.channels.keys())[key]]

    def __setitem__(self, key, value):
        if isinstance(value, (Channel, Event_Channel)):
            self.channels[key] = value

        else:
            raise ValueError("Value must be a Channel or Event_Channel object")
        return None

    def __repr__(self) -> str:
        representation = "Signal object with " + str(self.channel_count) + " channel(s)\n"
        for k, v in self.channels.items():
            representation += v.signal_name
            representation += " (" + str(v.sampling_rate) + "Hz)"
            # representation += " (" + str(v.signal_duration) + "s)"
            representation += " (" + str(v.n_windows) + " windows)"
            if isinstance(v, Channel):
                representation += " (" + str(v.channel.shape) + ")"
            representation += "\n"

        return representation

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Bio_Data):
            if self.get_channel_names() != other.get_channel_names():
                return False
            else:
                for k in self.get_channel_names():
                    if self[k] != other[k]:
                        return False
                else:
                    return True
        else:
            return False

    @property
    def channel_count(self):
        return len(self.channels)

    @property
    def multichannel(self):
        return self.channel_count > 1
