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
        
        self.timestamp_data = timestamp_data
        self.timestamp_resolution = timestamp_resolution
        self.indicator = indicator
        self.is_signal = is_signal
        self.sampling_rate = sampling_rate
        self.events = events
        self.event_name = event_name
        
        if(isinstance(event_name,str)):
            self.event_name=event_name
        else:
            raise ValueError("Event name must be a string")
        
        if(len(events) == 0):
            raise ValueError("Events cannot be empty")
        
        self._handle_events()
        

    def _handle_events(self):
        try:
            if(np.shape(self.events)[0]==1):
                self.windowed = False
            else:
                self.windowed = True
        except:
            raise ValueError("There is a problem creating with event. Event Channel name: "+self.event_name)
        
        if(self.is_signal):
            self._handle_signal_events()
            
        else:
            if(self.timestamp_data is None):
                raise ValueError("Timestamp data must be provided for event: "+self.event_name)
            if(self.windowed):
                timestamp_data = []
                for i in range(len(self.events)):
                    timestamp_data.append(self.timestamp_data[i][self.events[i]])
                self.timestamp_data = timestamp_data
            else:
                self.timestamp_data = self.timestamp_data[self.events]


    def _handle_signal_events(self):
        
        if(self.windowed):
            events=[]
            if(self.timestamp_data is None):
                raise ValueError("Timestamp data must be provided for windowed events")
            for i in range(len(self.events)):
                e = np.where(self.events[i]==self.indicator)
                events.append(e)
            self.events = events  
 
        elif(not self.windowed):
            events=[]
            if(self.timestamp_data is None):
                self.timestamp_data = timestamp_tools.create_timestamp_signal(self.timestamp_resolution,len(self.events),0,self.sampling_rate) 
            self.events = np.where(self.events==self.indicator)
            self.events = list(self.events)
            self.timestamp_data = list(self.timestamp_data[self.events])
                  
        else:
            raise ValueError("There is a problem creating with event. Event Channel name: "+self.event_name)

        