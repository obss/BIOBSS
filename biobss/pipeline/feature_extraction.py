from .bio_data import Bio_Data
import pandas as pd 
import numpy as np


class Feature():
    
    def __init__(self,name,function,parameters,input_signals):
        
        self_name=name
        self.function=function
        self.parameters=parameters
        self.input_signals=input_signals

        
    def run(self,data:Bio_Data)->pd.DataFrame:
        
        if(not isinstance(data,Bio_Data)):
            raise ValueError('Feature extraction must be run on a Bio_Data object')
        
        feature_output=pd.DataFrame()                
        data=data.copy()
        for channel_name in data.get_channel_names():
            self.parameters['prefix']=channel_name
            redundant = []
            for k in self.parameters:
                if k not in self.function.__code__.co_varnames:
                    redundant.append(k)
            for r in redundant:
                self.parameters.pop(r)
            if(channel_name in self.input_signals):
                timestamps=data[channel_name].get_timestamp()
                feature_set=[]
                for i,c in enumerate(data[channel_name].channel):
                    feature_set.append(self.function(c,**self.parameters))
                calculated_features=pd.DataFrame(feature_set,index=timestamps)
                
                feature_output=pd.concat([feature_output,calculated_features],axis=1)       
                
        return feature_output