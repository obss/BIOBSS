from .bio_data  import Bio_Data
from .data_channel import Data_Channel
import pandas as pd
import numpy as np


"""generic signal process object"""

class Bio_Process():
    
    def __init__(self,process_method,modality,sigtype,**kwargs) -> None:
        
        self.method=process_method
        self.modality=modality
        self.sigtype=sigtype
        self.kwargs=kwargs
        
        
    def check_modality(self,modality):
        if(self.modality==modality):
            return True
        else:
            return False
        
    def check_sigtype(self,sigtype):
        if(self.sigtype==sigtype):
            return True
        else:
            return False
        
    def process(self,signal:Data_Channel)->Data_Channel:
        signal=signal.copy()
        if(self.method.__annotations__['return']==Data_Channel):
            result=self.method(signal,**self.kwargs)
        else:
            self.kwargs.update({"sampling_rate":signal.sampling_rate})
            self.kwargs.update({"timestamp_start":signal.timestamp_start})
            self.kwargs.update({"name":signal.signal_name})
            redundant=[]
            for k in self.kwargs:
                if(k not in self.method.__code__.co_varnames):
                    redundant.append(k)
            for r in redundant:
                self.kwargs.pop(r)
            if(signal.shape[0]==1):
                result= self.method(signal.channel,**self.kwargs)
            else:
                output=[]
                for i in range(signal.shape[0]):
                    #Try vectorized method                        
                    output.append(self.method(signal.channel[i],**self.kwargs))
                    #If vectorized method fails, try scalar method
                    #if output is list of ndarrays, convert to single ndarray
                    #if output is list of lists, convert to single list
                    #if otput is list of pandas dataframes, convert to single pandas dataframe
                    
        if(isinstance(result,Data_Channel)):
            return result
        elif(isinstance(result,pd.DataFrame)):
            output=Bio_Data()
            for column in result.columns:
                output.add_channel(result[column],channel_name=column,sampling_rate=signal.sampling_rate,timestamp=signal.timestamp,timestamp_start=signal.timestamp_start)
            return output
        elif(isinstance(result,pd.Series)):
            return Data_Channel(result.values,name=result.name,sampling_rate=signal.sampling_rate,timestamp=signal.timestamp,timestamp_start=signal.timestamp_start)
        elif(isinstance(result,np.ndarray)):
            result=Data_Channel(result,sampling_rate=signal.sampling_rate,name=signal.signal_name,timestamp=signal.timestamp,timestamp_start=signal.timestamp_start)
            return result
        elif(isinstance(result,list)):
            result=Data_Channel(result,sampling_rate=signal.sampling_rate,name=signal.signal_name,timestamp=signal.timestamp,timestamp_start=signal.timestamp_start)
        else:
            raise ValueError('Result must be a Data_Channel, pd.DataFrame, pd.Series, np.ndarray or list')
    
    def get_name(self):
        return self.method.__name__

    
    

