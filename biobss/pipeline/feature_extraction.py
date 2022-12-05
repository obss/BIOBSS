from __future__ import annotations
from distutils.log import warn
from .bio_data import Bio_Data
import pandas as pd
import numpy as np
from typing import Union


class Feature():

    def __init__(self, name, function, input_signals: dict,parameters={}, **kwargs):

        self.name = name
        self.function = function
        self.parameters = parameters
        self.input_signals = input_signals
        self.kwargs=kwargs
        self.feature_output = pd.DataFrame()

    def run(self, data: Bio_Data) -> pd.DataFrame:

        self.parameters.update(self.kwargs)
        redundant = []
        for k in self.parameters:
            if k not in self.function.__code__.co_varnames:
                redundant.append(k)
        for r in redundant:
            self.parameters.pop(r)

        if(not isinstance(data, Bio_Data)):
            raise ValueError(
                'Feature extraction must be run on a Bio_Data object')

        self.data = data

        for input_prefix, input in self.input_signals.items():
            self.feature_output = pd.concat(
                [self.feature_output, self.__extract(input, input_prefix)], axis=1)
        return self.feature_output

    def __str__(self) -> str:
        return self.name

    def __extract(self, channel_name: Union[str, list], prefix: str = '') -> pd.DataFrame:
        if(isinstance(channel_name, str)):
            timestamps = self.data[channel_name].get_timestamp()
            feature_set = []
            self.parameters['prefix'] = prefix
            data = self.data[channel_name].channel
            
            if(len(data.shape) < 2):
                window_number = 1
            else:
                window_number = data.shape[0]       
            if(window_number == 1):
                feature_set.append(self.function(data, **self.parameters))
                calculated_features = pd.DataFrame.from_dict(feature_set)
                return calculated_features
            else:         
                for i in range(window_number):
                    feature_set.append(self.function(data[i], **self.parameters))
                calculated_features = pd.DataFrame(feature_set, index=timestamps)
                return calculated_features
            
            raise ValueError("Feature extraction failed unexpectedly!")
        elif(isinstance(channel_name, list)):
            ts = []
            for c in channel_name:
                ts.append(self.data[c].get_timestamp())

            ts = np.array(ts)
            if(np.any(ts-ts[0])):
                raise ValueError('All channels must have the same timestamp')

            timestamps = self.data[channel_name[0]].get_timestamp()
            feature_set = []
            self.parameters['prefix'] = prefix
            if(len(self.data[channel_name[0]].channel.shape) < 2):
                window_number = 1
            else:
                window_number = self.data[channel_name[0]].channel.shape[0]
            data_ = []
            for c in channel_name:
                data_.append(self.data[c].channel)
            data_ = np.array(data_)
            if(window_number!=1):
                for i in range(window_number):
                    feature_set.append(self.function(
                        data_[:, i], **self.parameters))
                calculated_features = pd.DataFrame(feature_set,index=timestamps)    
            else:
                timestamps=[timestamps[-1]]
                feature_set.append(self.function(
                        data_[:,], **self.parameters))

                calculated_features = pd.DataFrame.from_dict(feature_set)
                
            return calculated_features
