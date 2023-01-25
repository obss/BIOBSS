from __future__ import annotations
import numpy as np
import copy as copy
from numpy.typing import ArrayLike


class Channel():
    """ Biological signal channel class
    """
    def __init__(self, signal: ArrayLike, name: str, sampling_rate: float):

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
        self.signal_name = name
        self.sampling_rate = sampling_rate
                    
    def copy(self):
        return copy.deepcopy(self)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Channel):
            return False
        return ((self.signal_name == other.signal_name)
                and (np.array_equal(self.channel,other.channel)) and self.sampling_rate == other.sampling_rate)
        
    def get_window(self,window_index):
        if(len(self.channel.shape) < 2):
            return self.channel
        else:
            return self.channel[window_index,:]
        
    def get_timestamp(self):
        if(self.n_windows == 1):
            return np.array([0])
        else:
            return np.arange(self.n_windows)
    def get_window_timestamps(self):
        if(self.n_windows == 1):
            return np.array([0])
        else:
            return np.arange(self.n_windows)


    @property
    def duration(self):
        if(len(self.channel.shape) < 2):
            return self.channel.shape[0]/self.sampling_rate
        else:
            return self.channel.shape[1]/self.sampling_rate

    @property
    def n_windows(self):
        if(len(self.channel.shape) < 2):
            return 1
        else:
            return self.channel.shape[0]
        
    @property    
    def segmented(self):
        return (self.n_windows > 1)
    

