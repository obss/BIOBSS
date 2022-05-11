import numpy as np
import copy as copy
import pandas as pd
from numpy.typing import ArrayLike
from rsa import sign

class Signal():
    """ Signal object with add and iterate process objects"""
    def __init__(self,signal,sampling_rate,modality="Generic",sigtype="Generic",channel_names=[]):
        
        if(len(channel_names) != len(set(channel_names))): 
            raise ValueError('Channel names must be unique')
        if(signal.shape[0] != len(channel_names)):
            raise ValueError('Channel names must match signal dimensions')
        self.channel_count=len(channel_names)
        if(self.channel_count>1):
            self.multichannel=True
            
        self.data=pd.DataFrame(signal,columns=channel_names)
        self.sampling_rate=sampling_rate
        self.modality=modality
        self.sigtype=sigtype

    def add_channel(self,signal:ArrayLike,channel_name):

        if(channel_name in self.data.columns):
            raise ValueError('Channel name already exists')
        self.data[channel_name]=signal
        self.channel_count=self.channel_count+1
        if(self.channel_count>1):
            self.multichannel=True
        
    def add_multiple_channels(self,signal:ArrayLike,channel_names):
        if(len(channel_names) != len(signal)):
            raise ValueError('Channel names must match signal dimensions')
        if(len(channel_names)==0):
            raise ValueError('Channel names must be provided')
        if(channel_names == set(channel_names)):
            raise ValueError('Channel names must be unique')
        if(np.any(np.array(channel_names) in self.data.columns)):
            raise ValueError('One or more channel names already exist')
        
        for i,channel_name in enumerate(channel_names):
            self.data[channel_name]=signal[i]
            
        self.channel_count=self.channel_count+len(channel_names)
        if(self.channel_count>1):
            self.multichannel=True
            
    def remove_channel(self,channel_name):
        if(channel_name not in self.data.columns):
            raise ValueError('Channel does not exist')        
        self.data.drop(columns=channel_name,inplace=True)
        
    def modify_signal(self,signal:ArrayLike,channel_name):
        if(channel_name not in self.data.columns):
            raise ValueError('Channel does not exist')        
        self.data[channel_name]=signal
        
    def modify_multiple_signals(self,signal:ArrayLike,channel_names):
        if(len(channel_names) != len(signal)):
            raise ValueError('Channel names must match signal dimensions')
        if(len(channel_names)==0):
            raise ValueError('Channel names must be provided')
        if(channel_names == set(channel_names)):
            raise ValueError('Channel names must be unique')
        if(signal.shape[0] != len(channel_names)):
            raise ValueError('Channel names must match signal dimensions')
        
        for i,channel_name in enumerate(channel_names):
            self.data[channel_name]=signal[i]
            
    def copy(self):
        return copy.deepcopy(self)
    
    def __repr__(self) -> str:
        representation=self.modality+" Signal: "+self.sigtype+"\n"
        representation=representation+"Sampling Rate: "+str(self.sampling_rate)+"\n"
        representation=representation+"Channels: "+str(self.channel_count)+"\n"
        representation=representation+"Channel Names: "+str(self.channels)+"\n"
        representation=representation+"Multichannel: "+str(self.multichannel)+"\n"
        representation=representation+"Signal: "+str(self.signal)+"\n"
        return representation
        
