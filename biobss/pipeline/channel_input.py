from .bio_channel import Channel
import numpy as np
import pandas as pd
from .bio_data import Bio_Data
# self, signal: ArrayLike, name: str, sampling_rate: float,unit=None



def from_dict(signal_dict,sampling_rates,name = None,units=None):


    if not isinstance(signal_dict, dict):
        raise ValueError("signal_dict must be a dictionary while using from_dict function")
    
    if(name is not None):
        if(isinstance(name,str)):
            name = [name]
        elif(isinstance(name,list)):
            if(not all(isinstance(i, str) for i in name)):
                raise ValueError("name must be a list of strings")
            if(len(name) != len(signal_dict.keys())):
                raise ValueError("name must be the same length as signal_dict")
            name = name
        else:
            raise ValueError("name must be a list of strings or a string or None")

    output = Bio_Data()
    if(isinstance(sampling_rates,(int,float))):
        sampling_rates = [sampling_rates]
    elif(isinstance(sampling_rates,list)):
        if(not all(isinstance(i, (int,float)) for i in sampling_rates)):
            raise ValueError("sampling_rates must be a list of floats or integers")
        if(len(sampling_rates) != len(signal_dict.keys())):
            raise ValueError("sampling_rates must be the same length as signal_dict")
    if(units is None):
        units = [None]*len(signal_dict.keys())
    elif(units is not None and not isinstance(units,list)):
        raise ValueError("units must be a list of strings or None")
    elif(len(units) != len(signal_dict.keys())):        
        raise ValueError("units must be the same length as signal_dict")
    
    output = Bio_Data()        
    for i,(k,v) in enumerate(signal_dict.items()):
        if(name is None):
            c_name  = k
        else:
            c_name = name[i]
        output.add_channel(Channel(v,c_name,sampling_rates[i],units[i]))
    return output




def from_array(signal_array,sampling_rate,name,unit=None):
    if(name is None):
        raise ValueError("name must be provided while using from_array function")
    return Channel(signal_array,name, sampling_rate, unit)

def from_tuple(signal_tuple,index,name,sampling_rate=None,unit=None):
    if(name is None):
        raise ValueError("name must be provided while using from_tuple function")
    if(not isinstance(signal_tuple,tuple)):
        raise ValueError("signal_tuple must be a tuple while using from_tuple function")
    
    
    if(isinstance(index,int)):
        if(not isinstance(name,str)):
            raise ValueError("name must be a string")
        return Channel(signal_tuple[index],name,sampling_rate,unit)  
    elif(isinstance(index,list)):
        output = Bio_Data()
        if(not isinstance(name,list)):
            raise ValueError("name must be a list of strings")
        if(len(index) != len(name)):
            raise ValueError("index and name must be the same length")
        if(not isinstance(sampling_rate,list)):
            raise ValueError("sampling_rate must be a list of floats")
        if(len(index) != len(sampling_rate)):
            raise ValueError("index and sampling_rate must be the same length")
        if(unit is None):
            unit = [None]*len(index)
        elif(not isinstance(unit,list)):
            raise ValueError("unit must be a list of strings or None")        
        for i in range(len(index)):
            if(not isinstance(name[i],str)):
                raise ValueError("name must be a list of strings")
        if(any(isinstance(i, list) for i in index)):
            raise ValueError("index must be a list of integers")
        for i in range(len(index)):
            output[name[i]] = Channel(signal_tuple[index[i]],name[i],sampling_rate[i],unit[i])    
    else:
        raise ValueError("index must be an integer or list of integers")
    
    return output


def from_pd_series(signal,sampling_rate,name=None,unit=None):
    if(not isinstance(signal,pd.Series)):
        raise ValueError("signal must be a pandas Series for from_pd_series function")
    
    if(name is None):
        name = signal.name
    else:
        if(not isinstance(name,str)):
            raise ValueError("name must be a string")
    if(not isinstance(sampling_rate,(int,float))):
        raise ValueError("sampling_rate must be a float or integer")
    
    return Channel(signal.values,name,sampling_rate,unit)
        


def from_pd_df(signal_df,sampling_rate,name=None,unit=None):
    
    if(not isinstance(signal_df,pd.DataFrame)):
        raise ValueError("signal_df must be a pandas DataFrame for from_pd_df function")
    
    columns = signal_df.columns
    if(name is None):
        name = columns

    if(len(columns) == 1):
        if(isinstance(name,list)):
            name = name
        elif(isinstance(name,str)):
            name = [name]
        else:
            raise ValueError("name must be a string or list of strings")
        if(isinstance(unit,list)):
            if(len(unit) != 1):
                raise ValueError("unit must be a string or list of strings with length 1")
            unit = unit
        elif(isinstance(unit,str)):
            unit = [unit]
        elif(unit is None):
            unit = [None] * len(columns)
        if(isinstance(sampling_rate,list)):
            if(len(sampling_rate) != 1):
                raise ValueError("sampling_rate must be a float or list of floats with length 1")
            sampling_rate = sampling_rate
        elif(isinstance(sampling_rate,(int,float))):
            sampling_rate = [sampling_rate]
        else:
            raise ValueError("sampling_rate must be a float or list of floats")
        
        return Channel(signal_df.iloc[:,0].values,name[0],sampling_rate[0],unit[0])
    
    elif(len(columns) > 1):
        if(isinstance(name,list)):
            if(len(name) != len(columns)):
                raise ValueError("name must be a list of strings with length equal to number of columns")
            name = name
        elif(isinstance(name,str)):
            name = [name] * len(columns)
        else:
            raise ValueError("name must be a string or list of strings")
        if(isinstance(unit,list)):
            if(len(unit) != len(columns)):
                raise ValueError("unit must be a string or list of strings with length equal to number of columns")
            unit = unit
        elif(isinstance(unit,str)):
            unit = [unit] * len(columns)
        elif(unit is None):
            unit = [None] * len(columns)
        if(isinstance(sampling_rate,list)):
            if(len(sampling_rate) != len(columns)):
                raise ValueError("sampling_rate must be a float or list of floats with length equal to number of columns")
            sampling_rate = sampling_rate
        elif(isinstance(sampling_rate,(int,float))):
            sampling_rate = [sampling_rate] * len(columns)
        else:
            raise ValueError("sampling_rate must be a float or list of floats")
        
        output = Bio_Data()
        for i in range(len(columns)):
            output[name[i]] = Channel(signal_df.iloc[:,i].values,name[i],sampling_rate[i],unit[i])
        return output

        
def convert_channel(signal,sampling_rate = None,name=None,unit=None,index = None):
    if(index is not None):
        try:
            signal = signal[index]
        except:
            raise ValueError("index must be a valid key")
    
    if(isinstance(signal,Channel)):
        output = Bio_Data()
        output.add_channel(signal)
        return output
    elif(isinstance(signal,Bio_Data)):
        return signal
    
    if(sampling_rate is not None):
        if(isinstance(signal,tuple)):
            return convert_channel(from_tuple(signal,0,name,sampling_rate,unit))
        elif(isinstance(signal,pd.Series)):
            return convert_channel(from_pd_series(signal,sampling_rate,name,unit))
        elif(isinstance(signal,pd.DataFrame)):
            return convert_channel(from_pd_df(signal,sampling_rate,name,unit))
        elif(isinstance(signal,np.ndarray)):
            return convert_channel(from_array(signal,sampling_rate,name,unit))
        elif(isinstance(signal,list)):
            if(any(isinstance(x,Channel) for x in signal)):
                output = Bio_Data()
                for i in range(len(signal)):
                    output.add_channel(signal[i],name[i])
                return output
            else:
                signal = np.array(signal)
            return convert_channel(from_array(signal,sampling_rate,name,unit))

        elif(isinstance(signal,dict)):
            return from_dict(signal,sampling_rate,name,unit)
        elif(signal is None):
            raise ValueError("signal must not be None for convert function")
        else:
            raise ValueError("signal must be a tuple, pandas Series, pandas DataFrame, numpy array, list, or Channel for convert function")
    else:
        raise ValueError("sampling_rate must not be None for convert function with inputs other than Channel or Bio_Data")
    