from codecs import replace_errors
import numpy as np
import copy as copy
import pandas as pd
from numpy.typing import ArrayLike
from .bio_channel import Bio_Channel
from typing import Union
import warnings


class Bio_Data:
    """Signal object with add and iterate process objects"""

    def __init__(self):

        self.data = {}  # Data_Channel objects

    def add_channel(
        self,
        signal: Union[ArrayLike, Bio_Channel],
        channel_name: str = None,
        sampling_rate: Union[int, float] = None,
        timestamp: ArrayLike = None,
        timestamp_start: Union[int, float] = 0,
        timestamp_resolution:str = None,
        modify_existed: bool = False,
    ):

        if channel_name in self.data.keys():
            if not modify_existed:
                raise ValueError(
                    "Channel already exists set modify_existed to True to modify")
            else:
                self.modify_signal(
                    signal, channel_name, sampling_rate, timestamp, timestamp_start, timestamp_resolution
                )
                return

        if isinstance(signal, Bio_Channel):
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
            channel = Bio_Channel(
                signal,
                sampling_rate=sampling_rate,
                timestamp=timestamp,
                timestamp_start=timestamp_start,
                name=channel_name,
                timestamp_resolution=timestamp_resolution,
            )
            self.data[channel_name] = channel

    def remove_channel(self, channel_name):

        if channel_name not in self.data.keys():
            raise ValueError("Channel does not exist")
        self.data.pop(channel_name)

    def modify_signal(
        self,
        signal: Union[ArrayLike, Bio_Channel],
        channel_name,
        sampling_rate=None,
        timestamp=None,
        timestamp_start=0,
        timestamp_resolution=None,
    ):
        if channel_name not in self.data.keys():
            raise ValueError("Channel does not exist")
        if isinstance(signal, Bio_Channel):
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
                        "Timestamp not provided, using existing timestamp, if this is not correct, please provide a timestamp!"
                    )
            else:
                if(timestamp_resolution is None):
                    raise ValueError("Timestamp resolution must be provided if timestamp is provided")
                self.data[channel_name] = Bio_Channel(
                    signal,
                    sampling_rate=sampling_rate,
                    timestamp=timestamp,
                    timestamp_start=timestamp_start,
                    name=channel_name,
                    timestamp_resolution=timestamp_resolution,
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
                    timestamp_resolution=other.data[channel_name].timestamp_resolution,
                )
            else:
                self.modify_signal(
                    signal=other.data[channel_name].channel,
                    channel_name=channel_name,
                    sampling_rate=other.data[channel_name].sampling_rate,
                    timestamp=other.data[channel_name].timestamp,
                    timestamp_start=other.data[channel_name].timestamp_start,
                    timestamp_resolution=other.data[channel_name].timestamp_resolution,
                )

        
        return self

    def __getitem__(self, key: Union[str, int]) -> Bio_Channel:
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
            representation += " (" + str(v.sampling_rate) + "Hz)"
            representation += " (" + str(v.signal_duration) + "s)"
            representation += " (" + str(v.windows) + " windows)"
            representation += " (" + str(v.channel.shape) + ")"
            representation += "\n"

        return representation

    @property
    def channel_count(self):
        return len(self.data)
    
    @property
    def multichannel(self):
        return (self.channel_count > 1)
