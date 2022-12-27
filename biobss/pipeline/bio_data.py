"""Refactor this class"""

from __future__ import annotations
import numpy as np
import copy as copy
import pandas as pd
from numpy.typing import ArrayLike
from .bio_channel import Bio_Channel
from typing import Union
import warnings
from .event_channel import Event_Channel

class Bio_Data:
    """ Biological signal data class"""

    def __init__(self):
        """Initialize a Bio_Data object"""

        self.data = {}  # Data_Channel and Event_Channel objects
        self.events = {}  # Event_Channel objects
        
    def add_event_channel(self, event_channel: Union[Event_Channel,ArrayLike], modify_existed: bool = False , **kwargs):
        """ Add an event channel to the Bio_Data object

        Args:
            event_channel (Event_Channel): Event channel object to be added
            modify_existed (bool, optional): Whether to modify the existed channel or create new channel. Defaults to False.

        Raises:
            ValueError: If event_channel is not an Event_Channel object
            ValueError: If event_channel name already exists
        """

        if(isinstance(event_channel,Event_Channel)):
            if(event_channel.event_name in self.events.keys()):
                if(not modify_existed):
                    raise ValueError("Event channel already exists set modify_existed to True to modify")
                else:
                    self.modify_event_channel(event_channel)
            else:
                self.events[event_channel.event_name] = event_channel
                
        else:
            event_name = kwargs.get('event_name',None)
            timestamp_data = kwargs.get('timestamp_data',None)
            timestamp_resolution = kwargs.get('timestamp_resolution','ms')
            indicator = kwargs.get('indicator',1)
            is_signal = kwargs.get('is_signal',False)
            sampling_rate = kwargs.get('sampling_rate',None)            
            self.events[event_name] = Event_Channel(event_channel,event_name,timestamp_data,timestamp_resolution,indicator,is_signal,sampling_rate)

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
        """ Add a channel to the Bio_Data object

        Args:
            signal (Union[ArrayLike, Bio_Channel]): Signal content of the channel
            channel_name (str, optional): Name of the channel. Defaults to None.
            sampling_rate (Union[int, float], optional): Sampling rate of the channel. Defaults to None.
            timestamp (ArrayLike, optional): Timestamp of the channel. Defaults to None.
            timestamp_start (Union[int, float], optional): Start timestamp of the channel. Defaults to 0.
            timestamp_resolution (str, optional): Timestamp resolution of the channel. Defaults to None.
            modify_existed (bool, optional): Whether to modify the existed channel or create new channel. Defaults to False.

        Raises:
            ValueError: If channel_name is None
            ValueError: If channel_name already exists
            ValueError: If signal is not a Bio_Channel object and timestamp is None
            ValueError: If signal is not a Bio_Channel object and signal shape is different than existing signal
            ValueError: If timestamp_resolution is None and timestamp is provided
            ValueError: If sampling_rate is None and signal is not a Bio_Channel object
        """

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
        """ Remove a channel from the Bio_Data object

        Args:
            channel_name (str): Name of the channel to be removed
        Raises:
            ValueError: If channel_name does not exist
        """

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
        """Modify a channel in the Bio_Data object

        Args:
            signal (Union[ArrayLike, Bio_Channel]): Signal content of the channel
            channel_name (str): Name of the channel
            sampling_rate (Union[int, float], optional): Sampling rate of the channel. Defaults to None.
            timestamp (ArrayLike, optional): Timestamp of the channel. Defaults to None.
            timestamp_start (Union[int, float], optional): Start timestamp of the channel. Defaults to 0.
            timestamp_resolution (str, optional): Timestamp resolution of the channel. Defaults to None.

        Raises:
            ValueError: If channel_name does not exist
            ValueError: If signal is not a Bio_Channel object and timestamp is None
            ValueError: If signal is not a Bio_Channel object and signal shape is different than existing signal
            ValueError: If timestamp_resolution is None and timestamp is provided
            ValueError: If sampling_rate is None and signal is not a Bio_Channel object
        """
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
        """ "Join two Bio_Data object
        Args:
            other (Bio_Data): Bio_Data object to be joined
        Returns:
            Bio_Data: Joined Bio_Data object
        """
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
                
        for channel_name in other.events.keys():
            if channel_name not in self.data.keys():
                self.add_event_channel(
                    event=other.events[channel_name].events,
                    channel_name=channel_name,
                    timestamp=other.events[channel_name].timestamp,
                    timestamp_start=other.events[channel_name].timestamp_start,
                    timestamp_resolution=other.events[channel_name].timestamp_resolution,
                )
            else:
                self.add_event_channel(
                    event=other.events[channel_name].events,
                    channel_name=channel_name,
                    timestamp=other.events[channel_name].timestamp,
                    timestamp_start=other.events[channel_name].timestamp_start,
                    timestamp_resolution=other.events[channel_name].timestamp_resolution,
                    modify_existed=True
                )

        
        return self

    def __getitem__(self, key: str) -> Bio_Channel:
        if(isinstance(key, str)):
            if(key in self.data.keys()):
                return self.data[key]
            elif(key in self.events.keys()):
                return self.events[key]
            else:
                raise ValueError("Channel: "+ key +" does not exist!")
            

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
        for k,v in self.events.items():
            representation += "Event: " + k + "\n"

            

        return representation

    @property
    def channel_count(self):
        return len(self.data)
    
    @property
    def multichannel(self):
        return (self.channel_count > 1)
