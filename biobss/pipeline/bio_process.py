from __future__ import annotations
from .bio_data import Bio_Data
from .bio_channel import Bio_Channel
import pandas as pd
import numpy as np
import warnings
import inspect
from .event_channel import Event_Channel

"""generic signal process object"""


class Bio_Process:

    def __init__(self, process_method, inplace=True, prefix=None, return_index=None, argmap={},returns_event=False,event_args={}, **kwargs):

        self.process_method = process_method
        self.kwargs = kwargs
        self.inplace = inplace
        self.return_index = return_index
        self.argmap = argmap
        self.prefix = prefix
        self.returns_event = returns_event
        self.event_args=event_args
        if(self.inplace):
            self.prefix = None
        else:
            self.prefix= prefix if prefix is not None else "processed_"

    def map_args(self, signal: Bio_Channel):
        for key in self.argmap.keys():
            self.kwargs[self.argmap[key]] = signal.get_attribute(key)

    def process_args(self, signal: Bio_Channel):
        self.kwargs.update({"sampling_rate": signal.sampling_rate})
        self.kwargs.update({"timestamp_start": signal.timestamp_start})
        self.kwargs.update({"timestamp": signal.timestamp})
        self.kwargs.update({"name": signal.signal_name})
        self.map_args(signal)        
        signature = inspect.signature(self.process_method)
        excess_args = []
        for key in self.kwargs.keys():
            if key not in signature.parameters.keys():
                excess_args.append(key)
        for e in excess_args:
            self.kwargs.pop(e)
        return self.kwargs

    def process(self, signal: Bio_Channel):
        self.process_args(signal)
        has_return=False
        if("return" in self.process_method.__annotations__):
            if self.process_method.__annotations__["return"] in [Bio_Channel,Bio_Data]:
                has_return = True                
        if(has_return):
            result = self.process_method(signal, **self.kwargs)   
        else:
            if signal.channel.ndim == 1:
                result = self.process_method(signal.channel, **self.kwargs)
            else:
                try:
                    result = np.apply_along_axis(
                        self.process_method, 1, signal.channel, **self.kwargs)
                except:
                    warnings.warn(
                        "Vectorized method failed. Trying scalar method. It may be significantly slower.")
                finally:
                    result = []
                    for i in range(signal.channel.shape[0]):
                        result.append(self.process_method(
                            signal.channel[i], **self.kwargs))

        result = self._process_results(signal, result)
        return result

    def _process_results(self, signal: Bio_Channel, result):
        if(self.returns_event):
            result=self._process_event_results(signal,result)
            return result
        if isinstance(result, Bio_Channel):
            return result
        elif isinstance(result, Bio_Data):
            return result
        elif isinstance(result, tuple):
            if(self.return_index is not None):
                result = result[self.returnindex]
            else:
                raise ValueError(
                    "If process method returns a tuple, returnindex must be specified!")
        elif isinstance(result, pd.DataFrame):
            output = Bio_Data()

            for c in result.columns:
                out_name = self.prefix+"_"+c if self.prefix is not None else c
                output.add_channel(result[c],
                                   channel_name=out_name,
                                   sampling_rate=signal.sampling_rate,
                                   timestamp=signal.timestamp,
                                   timestamp_start=signal.timestamp_start,
                                   timestamp_resolution=signal.timestamp_resolution)
            return output
         
        sampling_rate=0
        timestamp=0
        timestamp_start=0
        timestap_resolution=signal.timestamp_resolution       
        
        if(isinstance(result,(np.ndarray,list))):
            if(self.return_index is not None):
                result = result[self.return_index]
            out_signal=np.array(result)
            if(len(result)!=len(signal.channel)):
                sampling_rate=len(result)/signal.signal_duration
                timestamp_start=signal.timestamp[0]
                timestap_resolution=signal.timestamp_resolution
                warnings.warn("Inplace process method returned a different length signal. Sampling rate and timestamp will be updated.")
            else:
                sampling_rate=signal.sampling_rate
                timestamp=signal.timestamp
                timestamp_start=signal.timestamp_start
                timestap_resolution=signal.timestamp_resolution
        elif(isinstance(result,pd.Series)):
            out_signal=result.values
            out_name=result.name
            if(len(result)!=len(signal.channel)):
                sampling_rate=len(result)/signal.signal_duration
                timestamp_start=signal.timestamp[0]
                timestap_resolution=signal.timestamp_resolution
                warnings.warn("Inplace process method returned a different length signal. Sampling rate and timestamp will be updated.")
            else:
                sampling_rate=signal.sampling_rate
                timestamp=signal.timestamp
                timestamp_start=signal.timestamp_start
                timestap_resolution=signal.timestamp_resolution
        else:
            raise ValueError("Result must be a Data_Channel, pd.DataFrame, pd.Series, np.ndarray or list")
                
        output= Bio_Channel(out_signal,
                            name=self.prefix+out_name if self.prefix is not None else signal.signal_name,
                            sampling_rate=sampling_rate,
                            timestamp=timestamp,
                            timestamp_start=timestamp_start,
                            timestamp_resolution=timestap_resolution)
                
           
        return output
        
    def _process_event_results(self, signal, result):
        timestamp = self.event_args.get("timestamp", signal.timestamp)
        timestamp_resolution = self.event_args.get("timestamp_resolution", signal.timestamp_resolution)
        indicator = self.event_args.get("indicator", 1)
        is_signal = self.event_args.get("is_signal", False)
        sampling_rate = self.event_args.get("sampling_rate", signal.sampling_rate)

        dict_index = self.event_args.get("dict_index", None)
        # If result is already an Event_Channel, simply return it
        if isinstance(result, Event_Channel):
            return result

        # Extract the specified element from the result if it is a tuple, array, or list
        if isinstance(result, tuple):
            result = result[self.return_index]
        elif isinstance(result, (np.ndarray, list)):
            if self.return_index is not None:
                result = result[self.return_index]
            signal_name = self.event_args.get("signal_name", "Events_" + signal.signal_name)

        # If result is a dictionary, extract the specified element and use it as the event data
        if isinstance(result, dict):
            if(isinstance(self.return_index,str)):
                index=self.return_index
            elif(dict_index is not None):
                index=dict_index
            else:
                raise ValueError("If result is a dictionary, dict_index must be specified! in either in the event_args or as a return_index")
            result = result[index]
            signal_name = index

        # If result is a Pandas Series, extract the values and use the series name as the event name
        elif isinstance(result, pd.Series):
            result = result.values
            signal_name = self.event_args.get("signal_name", result.name)

        # If result is a Pandas DataFrame, extract the specified column and use the column name as the event name
        elif isinstance(result, pd.DataFrame):
            result = result[self.return_index].values
            signal_name = self.return_index
            

        # Create the Event_Channel object and return it
        return Event_Channel(result, event_name=signal_name, timestamp_data=timestamp, timestamp_resolution=timestamp_resolution,
                             indicator=indicator, is_signal=is_signal, sampling_rate=sampling_rate)

    def __str__(self) -> str:
        return self.process_method.__name__
