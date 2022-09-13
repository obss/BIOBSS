from .bio_data import Bio_Data
from .bio_channel import Bio_Channel
import pandas as pd
import numpy as np
import warnings

"""generic signal process object"""

class Bio_Process:
    def __init__(self,process_method,input_signal='all',output_signal='self',use_prefix=False,returnindex=None,**kwargs) -> None:

        self.process_method = process_method
        self.kwargs = kwargs
        self.returnindex = returnindex
        self.input_signal = input_signal
        self.output_signal = output_signal
        self.use_prefix = use_prefix
        
        
    def check_input(self, channel_name: str) -> bool:
        if(channel_name in self.input_signal):
            return True
        elif(self.input_signal=='all'):
            return True
        else:
            return False

    def process(self, signal: Bio_Channel) -> Bio_Channel:
        #copy signal to prevent changing original signal
        signal = signal.copy()        
        has_return=False
        #check if process method has Bio_Channel as return type
        if("return" in self.process_method.__annotations__):
            if self.process_method.__annotations__["return"] == Bio_Channel:
                has_return = True
        
        # if process method has Bio_Channel as return type, return the result directly   
        if(has_return):
            result = self.process_method(signal, **self.kwargs)
        else:
            # if process method has no return type, return the result as a Bio_Channel
            self.kwargs.update({"sampling_rate": signal.sampling_rate})
            self.kwargs.update({"timestamp_start": signal.timestamp_start})
            self.kwargs.update({"timestamp": signal.timestamp})
            self.kwargs.update({"name": signal.signal_name})
            redundant = []
            # remove redundant  or unused arguments
            for k in self.kwargs:
                if k not in self.process_method.__code__.co_varnames:
                    redundant.append(k)
            for r in redundant:
                self.kwargs.pop(r)
            if len(signal.channel.shape) == 1:
                result = self.process_method(signal.channel, **self.kwargs)
            else:
                result = []
                # Try vectorized method
                try:
                    method_v = np.vectorize(self.process_method)
                    result = method_v(signal.channel, **self.kwargs)
                except:
                    warnings.warn(
                        "Vectorized method failed. Trying scalar method. It may be significantly slower."
                    )
                finally:
                    for i in range(signal.channel.shape[0]):
                        result.append(self.process_method(
                            signal.channel[i], **self.kwargs))

                # If vectorized method fails, try scalar method
                if isinstance(result[0], Bio_Channel):
                    pass
                elif isinstance(result[0], pd.DataFrame):
                    result_ = {}
                    for c in result[0].columns:
                        result_[c] = [a[c].to_list() for a in result]
                    tmp = Bio_Data()
                    for k in result_.keys():
                        tmp.add_channel(
                            result_[k],
                            channel_name=k,
                            sampling_rate=signal.sampling_rate,
                            timestamp=signal.timestamp,
                            timestamp_start=signal.timestamp_start,
                            modality=signal.signal_modality,
                        )
                    result = tmp.copy()
                # if output is list of ndarrays, convert to single ndarray
                elif isinstance(result[0], np.ndarray):
                    result = np.array(result)
                # if output is list of lists, convert to single list
                elif isinstance(result[0], list):
                    result = list(zip(*result))
                # if otput is list of pandas dataframes, convert to single pandas dataframe

        result=self._process_result(signal, result)
        return result
                   
            
    def _process_result(self,signal, result):
        if isinstance(result, tuple):
            if(self.returnindex is not None):
                result = result[self.returnindex]
            else:
                raise ValueError("If process method returns a tuple, returnindex must be specified!")
        
        if isinstance(result, Bio_Channel):
            if(not self.output_signal=='self'):
                if(self.use_prefix):
                    result.signal_name = self.output_signal+'_'+result.signal_name
                else:
                    result.signal_name = self.output_signal
            return result
        elif isinstance(result, Bio_Data):
            if(not self.output_signal=='self'):
                if(self.use_prefix):
                    for c in result.channels:
                        c.signal_name = self.output_signal+'_'+c.signal_name
                else:
                    for c in result.channels:
                        c.signal_name = self.output_signal
            return result
        elif isinstance(result, pd.DataFrame):
            output = Bio_Data()
            if(not self.output_signal=='self'):
                if(self.use_prefix):
                    for c in result.columns:
                        output.add_channel(result[c],
                                           channel_name=self.output_signal+'_'+c,
                                           sampling_rate=signal.sampling_rate,
                                           timestamp=signal.timestamp,
                                           timestamp_start=signal.timestamp_start,
                                           modality=signal.signal_modality)
                else:
                    for c in result.columns:
                        output.add_channel(result[c],
                                           channel_name=self.output_signal,
                                           sampling_rate=signal.sampling_rate,
                                           timestamp=signal.timestamp,
                                           timestamp_start=signal.timestamp_start,
                                           modality=signal.signal_modality)
            for column in result.columns:
                output.add_channel(
                    result[column],
                    channel_name=column,
                    sampling_rate=signal.sampling_rate,
                    timestamp=signal.timestamp,
                    timestamp_start=signal.timestamp_start,
                    modality=signal.signal_modality,
                )
            return output
        elif isinstance(result, pd.Series):
            output = Bio_Channel(
                result.values,
                name=result.name,
                sampling_rate=signal.sampling_rate,
                timestamp=signal.timestamp,
                timestamp_start=signal.timestamp_start,
                modality=signal.signal_modality,
            )
            if(not self.output_signal=='self'):
                if(self.use_prefix):
                    output.signal_name = self.output_signal+'_'+output.signal_name
                else:
                    output.signal_name = self.output_signal
            return output
        elif isinstance(result, np.ndarray):
            output = Bio_Channel(
                result,
                sampling_rate=signal.sampling_rate,
                name=signal.signal_name,
                timestamp=signal.timestamp,
                timestamp_start=signal.timestamp_start,
                modality=signal.signal_modality,
            )
            if(not self.output_signal=='self'):
                if(self.use_prefix):
                    output.signal_name = self.output_signal+'_'+output.signal_name
                else:
                    output.signal_name = self.output_signal
            return output
        elif isinstance(result, list):
            result = Bio_Channel(
                result,
                sampling_rate=signal.sampling_rate,
                name=signal.signal_name,
                timestamp=signal.timestamp,
                timestamp_start=signal.timestamp_start,
                modality=signal.signal_modality,
            )
            if(not self.output_signal=='self'):
                if(self.use_prefix):
                    output.signal_name = self.output_signal+'_'+output.signal_name
                else:
                    output.signal_name = self.output_signal            
            return result
        elif isinstance(result, dict):
            output = Bio_Data()
            for k in result.keys():
                output.add_channel(
                    result[k],
                    channel_name=k,
                    sampling_rate=signal.sampling_rate,
                    timestamp=signal.timestamp,
                    timestamp_start=signal.timestamp_start,
                    modality=signal.signal_modality,
                )
            if(not self.output_signal=='self'):
                if(self.use_prefix):
                    output.signal_name = self.output_signal+'_'+output.signal_name
                else:
                    output.signal_name = self.output_signal
            return output
        else:
            raise ValueError(
                "Result must be a Data_Channel, pd.DataFrame, pd.Series, np.ndarray or list"
            )
        

    def __str__(self) -> str:
        return self.process_method.__name__
