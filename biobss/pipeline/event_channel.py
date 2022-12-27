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
        self.channel = events
        self.event_name = event_name
        self.window_timestamps = None
        
        if(isinstance(event_name,str)):
            self.event_name=event_name
        else:
            raise ValueError("Event name must be a string")
        
        if(len(events) == 0):
            raise ValueError("Events cannot be empty")
        
        self._handle_events()
        

    def _handle_events(self):
        try:
            if(np.ndim(self.channel)<2):
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
                window_timestamps = []
                for i in range(len(self.channel)):
                    window_timestamps.append([self.timestamp_data[i][0],self.timestamp_data[i][-1]])
                    timestamp_data.append(self.timestamp_data[i][self.channel[i]])
                self.timestamp_data = timestamp_data
                self.window_timestamps = window_timestamps
            else:
                self.timestamp_data = self.timestamp_data[self.channel]
                self.window_timestamps = [self.timestamp_data[0],self.timestamp_data[-1]]
                
        


    def _handle_signal_events(self):
        
        if(self.windowed):
            events=[]
            if(self.timestamp_data is None):
                raise ValueError("Timestamp data must be provided for windowed events")
            for i in range(len(self.channel)):
                e = np.where(self.channel[i]==self.indicator)
                events.append(e)
            self.channel = events  
 
        elif(not self.windowed):
            events=[]
            if(self.timestamp_data is None):
                self.timestamp_data = timestamp_tools.create_timestamp_signal(self.timestamp_resolution,len(self.channel),0,self.sampling_rate) 
            self.channel = np.where(self.channel==self.indicator)
            self.channel = list(self.channel)
            self.timestamp_data = list(self.timestamp_data[self.channel])
                  
        else:
            raise ValueError("There is a problem creating with event. Event Channel name: "+self.event_name)
        
    def get_timestamp(self, ts_point="start") -> np.ndarray:

        if(not self.windowed):
            out = self.timestamp_data[0]
        else:
            if(not ts_point in ["start", "end", "mid"]):
                raise ValueError(
                    'ts_point must be "start","end","mid", Please specify a valid timestamp point')
            if(ts_point == "start"):
                out = np.array(self.window_timestamps)[:,0]
            elif(ts_point == "end"):
                out = np.array(self.window_timestamps)[:,1]
            elif(ts_point == "mid"):
                out = np.array(self.window_timestamps)[:,:].mean(axis=1)
        return out
        
        
    @property
    def windows(self):
        if(self.windowed):
            return len(self.channel)
        else:
            return 1

        