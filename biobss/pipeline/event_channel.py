from __future__ import annotations
import numpy as np
import copy as copy
from numpy.typing import ArrayLike
from ..timetools import timestamp_tools



class Event_Channel():
    
    """ Event channel class
            For storing event data, with the following properties:
    """
    
    
    def __init__(self, events: ArrayLike,timestamp, name: str,event_names=None,create_from_signal=False,indicators=[1],verbose=False):
        
        
        if(event_names is None):
            event_names = [name]        
        elif(isinstance(event_names,str)):
            event_names = [event_names]
        elif(not isinstance(event_names, list)):
            raise ValueError('event_names must be None or a list or a string')
            
        self.event_names=event_names
                
        if(create_from_signal):
            self._create_from_signal(events, indicators)
            
        else:
            for event_name in event_names:
                if(event_name in events):
                    raise ValueError('Event name already exists in events')
                else:
                    self.events[event_name] = events[event_name]
        
        timestamp = np.array(timestamp)
        
    def _create_from_signal(self,events, indicators=[1]):
            
        if(len(self.event_names) != len(indicators)):
            raise ValueError('event_names and indicators must have the same length') 
        self.event_indicators = indicators         
        self.events = {}
        for i in range(len(indicators)):
            self.events[self.event_names[i]]=np.where(events == indicators[i])[0]
            
            
    def _from_dict():
        pass
    
    def _from_list():
        pass
    
    def _from_signal():
        pass
        
        
        
    
    