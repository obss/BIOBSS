from __future__ import annotations
from .bio_data import Bio_Data
from .bio_channel import Channel
import pandas as pd
import numpy as np
import warnings
import inspect
from .event_channel import Event_Channel
from .channel_input import *
from .event_input import *

"""generic signal process object"""


class Bio_Process:

    def __init__(self, process_method,process_name, return_index=None, argmap={},returns_event=False,event_args={},*args, **kwargs):

        self.process_method = process_method
        self.args=args
        self.kwargs = kwargs
        self.return_index = return_index
        self.argmap = argmap
        self.returns_event = returns_event
        self.event_args=event_args
        self.name = process_name
        

    def map_args(self, signal: Channel,kwargs):
        for key in self.argmap.keys():
            kwargs[self.argmap[key]] = signal.get_attribute(key)
        return kwargs

    def process_args(self,args,kwargs):
        for a in args:
            if(isinstance(a,Channel)):
                kwargs.update({"sampling_rate": a.sampling_rate})
                kwargs.update({"name": a.channel_name})
                kwargs=self.map_args(a,kwargs)    
            elif(isinstance(a,Event_Channel)):
                kwargs.update({"sampling_rate": a.sampling_rate})
                kwargs.update({"event_args": self.event_args})
                kwargs.update({"name": a.channel_name})       
                kwargs=self.map_args(a,kwargs)    
        
        for v in kwargs.values():
            if(isinstance(v,Channel)):
                kwargs.update({"sampling_rate": v.sampling_rate})
                kwargs.update({"name": v.channel_name})
                kwargs=self.map_args(v,kwargs)
            elif(isinstance(v,Event_Channel)):
                kwargs.update({"sampling_rate": v.sampling_rate})
                kwargs.update({"event_args": self.event_args})
                kwargs.update({"name": v.channel_name})
                kwargs=self.map_args(v,kwargs)
                
        
        signature = inspect.signature(self.process_method)
        excess_args = []
        for key in kwargs.keys():
            if key not in signature.parameters.keys():
                excess_args.append(key)
        for e in excess_args:
            kwargs.pop(e)
        return kwargs

    def process(self,*args, **kwargs):
        kwargs.update(self.kwargs)
        args=args+self.args
        org_durations = []
        unit = []
        kwargs=self.process_args(args,kwargs)
        has_return=False
        if("return" in self.process_method.__annotations__):
            if self.process_method.__annotations__["return"] in [Channel,Bio_Data]:
                has_return = True                
        if(has_return):
            result = self.process_method(*args, **kwargs)
            #### TODO refactor args 
        else:
            if signal.channel.ndim == 1:
                result = self.process_method(*args, **kwargs)
                org_durations = [len(signal.channel.data)]
                unit = [signal.unit]
            else:
                try:
                    result = np.apply_along_axis(
                        self.process_method, 1,*args, **kwargs)
                except:
                    warnings.warn(
                        "Vectorized method failed. Trying scalar method. It may be significantly slower.")
                    result = []
                    for i in range(signal.channel.shape[0]):
                        result.append(self.process_method(
                            signal.channel[i],*args, **kwargs))
                        org_durations.append(len(signal.channel[i]))
                        unit.append(signal.channel.unit)

        if(self.returns_event):
            result = convert_event(result,signal,self.event_args,org_duration=org_durations,unit=unit)
        else:
            result = convert_channel(result,)
        return result

   
        

    def __str__(self) -> str:
        return self.process_method.__name__
