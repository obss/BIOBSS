from time import time
from venv import create
import matplotlib.pyplot as plt
from ..timetools import *
from ..plottools import *
from plotly.subplots import make_subplots
from plotly_resampler import register_plotly_resampler


def eda_plot(signals:dict,sampling_rate,timestamps=None,timestamp_resolution=None,figsize=(18.5, 10.5),show_peaks=True):
    
    
    eda_raw=signals.get('Raw')
    eda_cleaned=signals.get('Cleaned')
    tonic=signals.get('Tonic')
    phasic=signals.get('Phasic')
    peaks=signals.get('Peaks')
    
    if(timestamps is not None):
        timestamp=timestamps
        if(len(timestamps)!=len(eda_raw)):
            raise ValueError('Timestamps and EDA signal must have the same length')
        if(timestamp_resolution is None):
            raise ValueError('Timestamp resolution must be provided if timestamps are provided')
        else:
            timestamp_resolution=timestamp_resolution
    else:
        if(timestamp_resolution is None):
            timestamp_resolution='ms'
        timestamp=create_timestamp_signal(resolution=timestamp_resolution,length=len(eda_raw),rate=sampling_rate,start=0)
    
    # Create figure    
    fig, axs = plt.subplots(3)
    fig.set_size_inches(*figsize)
    
    # Plot raw EDA and Cleaned EDA
    if(eda_raw is not None):
        create_signal_plot(eda_raw,axs[0],timestamp=timestamp,timestamp_resolution=timestamp_resolution,plot_title='Raw EDA',signal_name='Raw')
    if(eda_cleaned is not None):
        create_signal_plot(eda_cleaned,axs[0],timestamp=timestamp,timestamp_resolution=timestamp_resolution,plot_title='Raw and Cleaned EDA',signal_name='Cleaned EDA')
    # Plot tonic EDA
    if(tonic is not None):
        create_signal_plot(tonic,axs[1],timestamp=timestamp,timestamp_resolution=timestamp_resolution,plot_title='Tonic EDA',signal_name='Tonic')

    # Plot phasic EDA and peaks
    if(phasic is not None):
        create_signal_plot(phasic,axs[2],timestamp=timestamp,timestamp_resolution=timestamp_resolution,plot_title='Phasic EDA',signal_name='Phasic',peaks=peaks)

    fig.tight_layout()
    
    plt.show()



def eda_plot_plotly(signals:dict,sampling_rate,timestamps=None,timestamp_resolution=None,figsize=(18.5, 10.5),show_peaks=True):
    
    
    eda_raw=signals.get('Raw')
    eda_cleaned=signals.get('Cleaned')
    tonic=signals.get('Tonic')
    phasic=signals.get('Phasic')
    peaks=signals.get('Peaks')
    # register_plotly_resampler(mode='auto')
    if(timestamps is not None):
        timestamp=timestamps
        if(len(timestamps)!=len(eda_raw)):
            raise ValueError('Timestamps and EDA signal must have the same length')
        if(timestamp_resolution is None):
            raise ValueError('Timestamp resolution must be provided if timestamps are provided')
        else:
            timestamp_resolution=timestamp_resolution
    else:
        if(timestamp_resolution is None):
            timestamp_resolution='ms'
        timestamp=create_timestamp_signal(resolution=timestamp_resolution,length=len(eda_raw),rate=sampling_rate,start=0)
    
    # Create figure    
    fig = make_subplots(rows=3, cols=1)    
    # Plot raw EDA and Cleaned EDA
    if(eda_raw is not None):
        create_signal_plot_plotly(eda_raw,fig,timestamp=timestamp,timestamp_resolution=timestamp_resolution,plot_title='Raw EDA',signal_name='Raw',location=(1,1))
    if(eda_cleaned is not None):
        create_signal_plot_plotly(eda_cleaned,fig,timestamp=timestamp,timestamp_resolution=timestamp_resolution,plot_title='Raw and Cleaned EDA',signal_name='Cleaned EDA',location=(1,1))
    # Plot tonic EDA
    if(tonic is not None):
        create_signal_plot_plotly(tonic,fig,timestamp=timestamp,timestamp_resolution=timestamp_resolution,plot_title='Tonic EDA',signal_name='Tonic',location=(2,1))

    # Plot phasic EDA and peaks
    if(phasic is not None):
        create_signal_plot_plotly(phasic,fig,timestamp=timestamp,timestamp_resolution=timestamp_resolution,plot_title='Phasic EDA',signal_name='Phasic',peaks=peaks,location=(3,1))

    fig.show()

