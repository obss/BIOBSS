from codecs import replace_errors
import numpy as np
import copy as copy
import pandas as pd
from numpy.typing import ArrayLike
from .data_channel import Data_Channel
from typing import Union
import warnings


class Bio_Data:
    """Signal object with add and iterate process objects"""

    def __init__(self):

        self.data = {}  # Data_Channel objects
        self.multichannel = False

    def add_channel(
        self,
        signal: Union[ArrayLike, Data_Channel],
        channel_name="Generic",
        sampling_rate: Union[int, float] = None,
        timestamp: ArrayLike = None,
        timestamp_start: Union[int, float] = 0,
        modality: str = "Generic",
        modify_existed: bool = False,
    ):

        if channel_name in self.data.keys():
            if not modify_existed:
                raise ValueError(
                    "Channel already exists set modify_existed to True to modify")
            else:
                self.modify_signal(
                    signal, channel_name, sampling_rate, timestamp, timestamp_start
                )
                return

        if isinstance(signal, Data_Channel):
            self.data[signal.signal_name] = signal.copy()
        else:
            signal = np.array(signal)
            if sampling_rate is None:
                raise ValueError(
                    "Sampling rate must be provided if signal is not a Data_Channel object"
                )
            if channel_name is None:
                raise ValueError(
                    "Channel name must be provided if signal is not a Data_Channel object"
                )
            channel = Data_Channel(
                signal,
                sampling_rate=sampling_rate,
                timestamp=timestamp,
                timestamp_start=timestamp_start,
                name=channel_name,
                modality=modality,
            )
            self.data[channel_name] = channel

        if self.channel_count > 1:
            self.multichannel = True

    def remove_channel(self, channel_name):

        if channel_name not in self.data.keys():
            raise ValueError("Channel does not exist")
        self.data.pop(channel_name)
        if self.channel_count == 1:
            self.multichannel = False

    def modify_signal(
        self,
        signal: Union[ArrayLike, Data_Channel],
        channel_name,
        sampling_rate=None,
        timestamp=None,
        timestamp_start=0,
        modality="Generic",
    ):
        if channel_name not in self.data.keys():
            raise ValueError("Channel does not exist")
        if isinstance(signal, Data_Channel):
            self.data[channel_name] = signal.copy()
        else:
            if timestamp is None:
                if self.data[channel_name].channel.shape != signal.shape:
                    raise ValueError(
                        "Timestamp must be provided if signal is not a Data_Channel object and shape is different than existing signal"
                    )
                else:
                    self.data[channel_name].channel = signal
                    warnings.warn(
                        "Timestamp not provided, using existing timestamp, if this is not correct, please provide timestamp"
                    )
            else:
                self.data[channel_name] = Data_Channel(
                    signal,
                    sampling_rate=sampling_rate,
                    timestamp=timestamp,
                    timestamp_start=timestamp_start,
                    name=channel_name,
                    modality=modality,
                )

    def get_channel_names(self):
        return list(self.data.keys())

    def copy(self):
        return copy.deepcopy(self)

    def join(self, other: "Bio_Data"):
        """Join two Bio_Data objects"""
        for channel_name in other.data.keys():
            if channel_name not in self.data.keys():
                self.add_channel(
                    signal=other.data[channel_name].channel,
                    channel_name=channel_name,
                    sampling_rate=other.data[channel_name].sampling_rate,
                    timestamp=other.data[channel_name].timestamp,
                    timestamp_start=other.data[channel_name].timestamp_start,
                    modality=other.data[channel_name].signal_modality,
                )
            else:
                self.modify_signal(
                    signal=other.data[channel_name].channel,
                    channel_name=channel_name,
                    sampling_rate=other.data[channel_name].sampling_rate,
                    timestamp=other.data[channel_name].timestamp,
                    timestamp_start=other.data[channel_name].timestamp_start,
                    modality=other.data[channel_name].signal_modality,
                )

        return self

    def __getitem__(self, key: Union[str, int]) -> Data_Channel:
        if(isinstance(key, str)):
            return self.data[key]
        elif(isinstance(key, int)):
            return self.data[list(self.data.keys())[key]]

    def __setitem__(self, key, value):
        return self.modify_signal(value, key)

    def __repr__(self) -> str:
        representation = (
            "Signal object with " + str(self.channel_count) + " channel(s)\n"
        )
        for k, v in self.data.items():
            representation += v.signal_name
            representation += " (" + v.signal_modality + ")"
            representation += " (" + str(v.sampling_rate) + "Hz)"
            representation += " (" + str(v.signal_duration) + "s)"
            representation += " (" + str(v.windows) + " windows)"
            representation += " (" + str(v.channel.shape) + ")"
            representation += "\n"

        return representation

    @property
    def channel_count(self):
        return len(self.data)
