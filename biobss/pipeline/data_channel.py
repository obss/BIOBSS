import numpy as np
import copy as copy
from numpy.typing import ArrayLike



class Data_Channel():
    """ Signal object with add and iterate process objects"""
    def __init__(self,signal:ArrayLike,sampling_rate:float,timestamp=None,timestamp_start=0,name="Generic",modality="Generic"):
        
        regularity_parameter=0.01        
        self.channel=np.array(signal)
        if(sampling_rate<0):
            raise ValueError('Sampling rate must be greater than 0')
        
        self.sampling_rate=sampling_rate
        if(timestamp is not None):
            timestamp=np.array(timestamp)
            if(timestamp.shape != signal.shape):
                raise ValueError(timestamp.shape,signal.shape,'Timestamp must match signal dimensions')
            if(np.any(np.diff(timestamp)<0)):
                raise ValueError('Timestamp must be monotonic')
            if(np.diff(timestamp).std()>regularity_parameter):# TODO: optimize this parameter
                raise ValueError('Timestamp must be regularly spaced')
            self.timestamp=timestamp
        else:
            if(timestamp_start<0):
                raise ValueError('Timestamp start must be greater than 0')
            timestamp=np.arange(len(signal))/sampling_rate
            self.timestamp=timestamp+timestamp_start
        
        self.timestamp_start = timestamp_start    
        self.signal_name=name
        self.signal_modality=modality
        self.signal_duration=len(signal)/sampling_rate
        
    def copy(self):
        return copy.deepcopy(self)
    
    def change_channel_data(self,signal:ArrayLike):
        self.channel=np.array(signal)
    
    
    def __str__(self) -> str:
        return str(self.channel)
    def __repr__(self) -> str:
        representation = self.signal_name
        representation += " (" + self.signal_modality + ")"
        representation += " (" + str(self.sampling_rate) + "Hz)"
        representation += " (" + str(self.signal_duration) + "s)"
        representation += " (" + str(self.channel.shape) + ")"
        representation += " (" + str(self.timestamp.shape) + ")"
        representation += " (" + str(self.channel.dtype) + ")"
        representation += self.__str__()
        return representation
    
