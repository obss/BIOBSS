from __future__ import annotations
import numpy as np
import copy as copy
from numpy.typing import ArrayLike
from ..timetools import timestamp_tools
import pandas as pd


class Event_Channel():

    """ Event channel class
            For storing event data, with the following properties:
    """
    # Event name
    # Event data
    # Timestamp Data

    def __init__(self, events:ArrayLike, event_name, timestamp_data=None, timestamp_resolution='ms', indicator=1, is_signal=False, sampling_rate=None):
        self.event_names = event_name
        if(len(events) == 0):
            raise ValueError("Events cannot be empty")
        elif(len(np.shape(events)) > 1):
            raise ValueError("Events must be a 1D array")

        if(is_signal):
            if(timestamp_data is None):
                if(sampling_rate is None):
                    raise ValueError("Sampling rate cannot be empty if timestamp data is not provided")
                timestamp_data = timestamp_tools.get_timestamps(np.arange(len(
                    events)), sampling_rate=sampling_rate, timestamp_resolution=timestamp_resolution)

            self.events = np.where(events == indicator)[0]
            self.timestamp_data = timestamp_data[self.events]

        else:
            if(timestamp_data is None):
                raise ValueError(
                    "Timestamp data cannot be empty if events are not signals")

            if(len(events) != len(timestamp_data)):
                try:
                    timestamp_data = timestamp_data[events]
                except IndexError:
                    raise IndexError(
                        "Events and timestamp data must have the same length or can be indexed from timestamp data with event data")

            self.timestamp_data = timestamp_data
            self.events = events

        self.content=pd.Series(self.events,index=self.timestamp_data,name=self.event_names)
            

