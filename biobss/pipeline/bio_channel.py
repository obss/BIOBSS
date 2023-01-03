from __future__ import annotations
import numpy as np
import copy as copy
from numpy.typing import ArrayLike
from ..timetools import timestamp_tools


class Bio_Channel():
    """ Biological signal channel class
    """
    def __init__(self, signal: ArrayLike, name: str, sampling_rate: float, timestamp=None, timestamp_resolution=None, timestamp_start=0, verbose=False,unit=None):

        # Docstring
        """ Biological signal channel class
        Args:
            signal (ArrayLike): signal data
            name (str): signal name
            sampling_rate (float): signal sampling rate
            timestamp (ArrayLike): signal timestamp
            timestamp_resolution (float): signal timestamp resolution
            timestamp_start (float): signal timestamp start
            verbose (bool): print debug info
            unit (str): signal unit
        
        returns:
            Bio_Channel: Bio_Channel object
        
        raises:
            ValueError: if signal and timestamp dimensions do not match
            ValueError: if signal and timestamp resolution do not match
        # End Docstring        
        """

        # initialize channel data
        self.channel = np.array(signal)
        self.timestamp_resolution = timestamp_resolution
        self.verbose = verbose
        self.signal_name = name

        self._initialize_sampling_rate(sampling_rate)        
        self._initialize_timestamp(timestamp,timestamp_resolution, timestamp_start, sampling_rate)

        # initialize signal duration and windows
        if(len(signal.shape) < 2):
            self.signal_duration = len(signal)/sampling_rate,
            self.windows = 1
        else:
            self.signal_duration = signal.shape[1]/sampling_rate
            self.windows = signal.shape[0]
            
    def _initialize_sampling_rate(self, sampling_rate: float):
        if(sampling_rate < 0):
            raise ValueError('Sampling rate must be greater than 0')
        self.sampling_rate = sampling_rate
            
    def _initialize_timestamp(self,timestamp, timestamp_resolution: str, timestamp_start: float,sampling_rate: float):
        if(timestamp is not None):
            timestamp = np.array(timestamp)
            if(timestamp_resolution is None):
                raise ValueError('Timestamp resolution must be provided if timestamp is provided')
            if(timestamp.shape != self.channel.shape):
                raise ValueError(timestamp.shape, self.channel.shape,
                                 'Timestamp must match signal dimensions')
            if(timestamp_tools.check_timestamp(timestamp, timestamp_resolution)):
                self.timestamp = timestamp
                if(self.verbose):
                    print("Input is set with timestamp resolution of " +
                          str(timestamp_resolution))
            else:
                raise ValueError('Timestamp is not valid')
        else:
            if(timestamp_resolution is None):
                timestamp_resolution = "ms"
                self.timestamp_resolution = timestamp_resolution
            if(self.channel.ndim > 1):
                raise ValueError(
                    'Timestamp must be provided for multi-channel signals')
            self.timestamp = timestamp_tools.create_timestamp_signal(
                timestamp_resolution, len(self.channel), timestamp_start, self.sampling_rate)
            self.timestamp_start = timestamp_start
            if(self.verbose):
                print("Input is set with timestamp resolution of " +
                      str(timestamp_resolution))
        self.timestamp_start = timestamp_start

    def copy(self):
        """Returns a copy of the Bio_Channel object

        Returns:
            Bio_Channel: Bio_Channel object
        """
        return copy.deepcopy(self)

    def sync_timestamps(self, offset: float):
        if(len(self.timestamp.shape) == 1):
            self.timestamp = self.timestamp+offset
        else:
            self.timestamp[:, 0] = self.timestamp[:, 0]+offset

    def change_channel_data(self, signal: ArrayLike):
        if(signal.shape != self.channel.shape):
            raise ValueError('New signal must match channel dimensions')
        self.channel = np.array(signal)

    def get_timestamp(self, ts_point="start") -> np.ndarray:

        if(len(self.timestamp.shape) == 1):
            out = np.array(self.timestamp)
        else:
            if(not ts_point in ["start", "end", "mid"]):
                raise ValueError(
                    'ts_point must be "start","end","mid", Please specify a valid timestamp point')
            if(ts_point == "start"):
                out = np.array(self.timestamp[:, 0])
            elif(ts_point == "end"):
                out = np.array(self.timestamp[:, -1])
            elif(ts_point == "mid"):
                out = np.array(
                    self.timestamp[:, int(len(self.timestamp[0])/2)])

        return out
    
    def get_window_timestamps(self, window = 0, ts_point="start") -> np.ndarray:

        if(self.windows == 1):
            return self.timestamp[0]
        if(len(self.timestamp.shape) == 1):
            out = np.array(self.timestamp)
        else:
            if(not ts_point in ["start", "end", "mid"]):
                raise ValueError(
                    'ts_point must be "start","end","mid", Please specify a valid timestamp point')
            if(ts_point == "start"):
                out = np.array([self.timestamp[window, 0]])
            elif(ts_point == "end"):
                out = np.array([self.timestamp[window, -1]])
            elif(ts_point == "mid"):
                out = np.array(
                    [self.timestamp[window, int(len(self.timestamp[window])/2)]])

        return out

    def __str__(self) -> str:
        return str(self.channel)
    
    def get_attribute(self, __name: str):
        if(__name == "channel"):
            return np.array(self.channel)
        elif(__name == "timestamp"):
            return np.array(self.timestamp)
        elif(__name == "sampling_rate"):
            return self.sampling_rate
        elif(__name == "signal_name"):
            return self.signal_name
        elif(__name == "signal_duration"):
            return self.signal_duration
        elif(__name == "windows"):
            return self.windows
        elif(__name == "timestamp_start"):
            return self.timestamp_start
        elif(__name == "timestamp_resolution"):
            return self.timestamp_resolution
        else:
            raise KeyError("Attribute not found")

    def __repr__(self) -> str:
        representation = self.signal_name
        representation += " (" + str(self.sampling_rate) + "Hz)"
        representation += " (" + str(self.signal_duration) + "s)"
        representation += " (" + str(self.windows) + " windows)"
        representation += " (" + str(self.channel.shape) + ")"
        representation += " (" + str(self.timestamp.shape) + ")"
        representation += " (" + str(self.channel.dtype) + ")"
        representation += self.__str__()
        return representation

