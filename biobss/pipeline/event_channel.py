from __future__ import annotations
import numpy as np
import copy as copy
from numpy.typing import ArrayLike


class Event_Channel():
    """ Biological signal channel class
    """
    def __init__(self, events: dict, name: str, sampling_rate: float,org_duration=None,unit=None):

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
        if(not isinstance(events, dict)):
            raise ValueError("events must be a dictionary")
        elif(len(events) == 0):
            raise ValueError("events must have at least one key")
        elif(not all(isinstance(key, str) for key in events.keys())): 
            raise ValueError("events keys must be strings")
        elif(not all(isinstance(value, (np.ndarray,list)) for value in events.values())):
            raise ValueError("events values must be lists or numpy arrays")
        elif(not isinstance(name, str)):
            raise ValueError("name must be a string")
        elif(not isinstance(sampling_rate, (float,int))):
            raise ValueError("sampling_rate must be a float or int")
        elif(unit is not None and not isinstance(unit, str)):
            raise ValueError("unit must be a string or None")            

        self.channel = events
        self.signal_name = name
        self.sampling_rate = sampling_rate
        self.org_duration=org_duration
                    
        if(unit is not None):
            self.unit = unit
        else:
            self.unit = "NA"
       

    def get_event(self, event_name: str):
        if(event_name in self.channel.keys()):
            return self.channel[event_name]
        else:
            raise ValueError("event_name not found in channel")
        
    def __getitem__(self, event_name: str):
        return self.get_event(event_name)
    
    def __setitem__(self, event_name: str, event: ArrayLike):
        if(not isinstance(event, (np.ndarray,list))):
            raise ValueError("event must be a list or numpy array")
        elif(not isinstance(event_name, str)):
            raise ValueError("event_name must be a string")
        else:
            self.channel[event_name] = np.array(event)
            
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Event_Channel):
            return NotImplemented
        return self.channel == other.channel and self.signal_name == other.signal_name and self.sampling_rate == other.sampling_rate and self.unit == other.unit
    
    def get_window(self,window: int,key=None):
        
        out = {}
        if key is None:
            for key in self.channel.keys():
                out[key] = self.channel[key][window]
        else:
            out[key] = self.channel[key][window]               

    def copy(self):
        return copy.deepcopy(self)
    
    def __copy__(self):
        return self.copy()

    @property
    def n_events(self):
        return len(self.channel.keys())
    
    @property
    def event_names(self):
        return list(self.channel.keys())

    @property
    def n_windows(self):
        return len(self.channel[self.event_names[0]])
        
    @property    
    def segmented(self):
        return (self.n_windows > 1)
    
    

