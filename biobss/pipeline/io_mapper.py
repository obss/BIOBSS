from .bio_channel import Bio_Channel
from .bio_data import Bio_Data
import pandas as pd



"""write a method to map given input signals to required input types of a process"""
def bio_channel_to_df(signal: Bio_Channel):
    """convert a Bio_Channel object to a pandas dataframe"""
    
    df = pd.DataFrame(signal.channel)
    df.columns = [signal.signal_name]
    return df




def output_mapper():
    """Map output signals to output channels of a process"""
    pass