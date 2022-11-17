from time import time
from .bio_data import Bio_Data
from .bio_channel import Bio_Channel
import pandas as pd
import numpy as np
import warnings
import inspect

"""generic signal process object"""


class Bio_Process:

    def __init__(self, process_method, inplace=True, prefix=None, return_index=None, argmap={}, **kwargs):

        self.process_method = process_method
        self.kwargs = kwargs
        self.inplace = inplace
        self.return_index = return_index
        self.argmap = argmap
        self.prefix = prefix
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
        


    def __str__(self) -> str:
        return self.process_method.__name__
