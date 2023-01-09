import numpy as np
import pandas as pd
from .bio_data import Bio_Data
from .event_channel import Event_Channel


def event_from_signal(signal,indicator):
    
    if(not isinstance(signal, (np.ndarray,list))):
        raise ValueError("signal must be a list or numpy array")
    
    signal = np.array(signal)
    event = np.where(signal == indicator)[0]  
    return event


def event_from_dict(event,sampling_rate,name=None,from_signal=False,indicator=None,org_duration=None,unit=None):
    if(not isinstance(event, dict)):
        raise ValueError("events must be a dictionary")
    if(from_signal):
        if(indicator is None):
            raise ValueError("indicator must be provided while using from_signal")
        for key in event.keys():
            event[key] = event_from_signal(event[key],indicator)

    if(name is None):
        name = event.keys()[0]
            
    return Event_Channel(event,name,sampling_rate,org_duration,unit)

def event_from_list(event,sampling_rate,name,from_signal=False,indicator=None,org_duration=None,unit=None):
    if(from_signal):
        if(indicator is None):
            raise ValueError("indicator must be provided while using from_signal")
        event = event_from_signal(event,indicator)
        
    event_dict = {name:event}    
    return Event_Channel(event_dict,name,sampling_rate,org_duration,unit)


def event_from_tuple(event,sampling_rate,name,index=None,from_signal=False,indicator=None,org_duration=None,unit=None):
    if(index is None and len(name)):
        pass
    if(not isinstance(index, (int,float))):
        raise ValueError("index must be a float or integer")
    if(not isinstance(sampling_rate, (int,float))):
        raise ValueError("sampling_rate must be a float or integer")
    if(not isinstance (name, str)):
        raise ValueError("name must be a string")
    if(from_signal):
        if(indicator is None):
            raise ValueError("indicator must be provided while using from_signal")
        event = event_from_signal(event,indicator)
        
    event_dict = {name:event}    
    return Event_Channel(event_dict,name,sampling_rate,org_duration,unit)

def event_from_dataframe(df, sampling_rate, name = None, index=None, from_signal=False, indicator=None ,org_duration=None,unit=None):

    if(from_signal):
        if(indicator is None):
            raise ValueError("indicator must be provided while using from_signal")
        else:
            df = df[df == indicator]
            
    if(name is None):
        name = df.columns[0]
        
    if(index is not None):
        df = df.iloc[index]
        
    event_dict = {}
    for col in df.columns:
        event_dict[col] = list(df[col].values)
        
    return Event_Channel(event_dict,name,sampling_rate,org_duration,unit)
            
def convert_event(event,sampling_rate= None,name = None,index=None,from_signal=False,indicator=None,org_duration=None,unit=None):
    if(isinstance(event,Event_Channel)):
        output = Bio_Data()
        output.add_channel(event)
        return output
    
    else:
        if(sampling_rate is None):
            raise ValueError("sampling_rate must be provided")
        if(name is None and not isinstance(event,dict)):
            raise ValueError("name must be provided if event is not an Event_Channel or dictionary")
        if(isinstance(event, (np.ndarray,list))):
            output = event_from_list(event,sampling_rate,name,from_signal,indicator,org_duration,unit)
        elif(isinstance(event, dict)):
            output = event_from_dict(event,sampling_rate,name,from_signal,indicator,org_duration,unit)
        elif(isinstance(event, tuple)):
            output = event_from_tuple(event,sampling_rate,name,index,from_signal,indicator,org_duration,unit)
        elif(isinstance(event, pd.DataFrame)):
            output = event_from_dataframe(event,sampling_rate,name,index,from_signal,indicator,org_duration,unit)
        else:
            raise ValueError("event must be a list, tuple, or dictionary")
    
    result = Bio_Data()
    result.add_channel(output)
    return result