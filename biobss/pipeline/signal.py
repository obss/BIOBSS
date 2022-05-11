import numpy as np
import copy as copy

class Signal():
    """ Signal object with add and iterate process objects"""
    def __init__(self,signal,sampling_rate,modality="Generic",sigtype="Generic",multichannel=False,channel_names=[]):
        signal=np.array(signal)
        self.signal=[signal]
        self.sampling_rate=sampling_rate
        self.modality=modality
        self.sigtype=sigtype
        
        if(multichannel):
            self.multichannel=True
            if(len(channel_names) != len(set(channel_names))):
                raise ValueError('Channel names must be unique')
            if(signal.shape[0] != len(channel_names)):
                raise ValueError('Channel names must match signal dimensions')
            self.channel_count=len(channel_names)
            self.channels=channel_names
            
        else:
            self.multichannel=False
            self.channel_count=1
            self.channels=[sigtype]


    def add_channel(self,signal,channel_name):
        if(self.channels.count(channel_name)>0):
            raise ValueError('Channel name already exists')
        signal=np.array(signal)
        self.signal.append(signal)
        self.channel_count=self.channel_count+1
        self.channels.append(channel_name)
        if(self.channel_count>1):
            self.multichannel=True
        
    def add_multiple_channels(self,signal,channel_names):
        if(len(channel_names) != len(signal)):
            raise ValueError('Channel names must match signal dimensions')
        for i in range(len(signal)):
            self.add_channel(signal[i],channel_names[i])
    
    def __getitem__(self,key):
        index=self.channels.index(key)
        return self.signal[index]
    
    def change_channel_name(self,old_name,new_name):
        if(self.channels.count(new_name)>0):
            raise ValueError('Channel name already exists')
        index=self.channels.index(old_name)
        self.channels[index]=new_name
        
    def change_channel_data(self,channel_name,signal):
        index=self.channels.index(channel_name)
        self.signal[index]=signal

    def copy(self):
        return copy.deepcopy(self)

    
    def get_channel_data(self,channel_name):
        index=self.channels.index(channel_name)
        return self.signal[index]
    
    def signal_process(self,process_method,**kwargs):
        """
        Processes the signal with the given method
        """
        signal=self.copy()
        kwargs.update({"sampling_rate":self.sampling_rate})
        kwargs.update({"modality":self.modality})
        kwargs.update({"sigtype":self.sigtype})
        kwargs.update({"multichannel":self.multichannel})
        kwargs.update({"channel_names":self.channels})
        for c in self.channels:
            signal.change_channel_data(c,process_method(self[c],**kwargs))

        return signal
    
    
    def __repr__(self) -> str:
        representation=self.modality+" Signal: "+self.sigtype+"\n"
        representation=representation+"Sampling Rate: "+str(self.sampling_rate)+"\n"
        representation=representation+"Channels: "+str(self.channel_count)+"\n"
        representation=representation+"Channel Names: "+str(self.channels)+"\n"
        representation=representation+"Multichannel: "+str(self.multichannel)+"\n"
        representation=representation+"Signal: "+str(self.signal)+"\n"
        return representation
        
