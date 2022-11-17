import matplotlib.pyplot as plt
import numpy as np
from ..timetools import *
from numpy.typing import ArrayLike
import plotly.graph_objects as go
from plotly_resampler import register_plotly_resampler


def create_signal_plot(signal,ax,timestamp=None,timestamp_resolution=None,plot_title="Signal Plot",signal_name="Signal",peaks=None):
    
    if(timestamp is None):
        if(timestamp_resolution is None):
            timestamp_resolution = 's'
        timestamp = timestamp_tools.create_timestamp_signal(timestamp_resolution,signal.size,0,1)
                       
    else:
        if(timestamp_resolution is None):
            raise ValueError('timestamp_resolution must be specified if timestamp is specified')
    

    # Check if there is existing legend
    if(isinstance(ax.get_legend(),type(None))):
        legend = []
    else:
        legend = [x.get_text() for x in ax.get_legend().texts]

    ax.plot(timestamp,signal)
    ax.set_title(plot_title)
    ax.set_xlabel('Time ('+timestamp_resolution+')')
    ax.set_ylabel('Amplitude')
    legend.append(signal_name)
    if(peaks is not None):
        ax.scatter(timestamp[peaks],signal[peaks],c='r')
        legend.append('Peaks')  
    ax.legend(legend)
    
    
def create_signal_plot_plotly(signal,fig,timestamp=None,timestamp_resolution=None,plot_title="Signal Plot",signal_name="Signal",peaks=None,location=None):
    
    #adjust it
    limit=200000
    
    if(len(signal)>limit):
        Warning('Signal is too large and will be resampled. Consider using create_signal_plot instead')
        register_plotly_resampler(mode='auto')
    if(location is None):
        raise ValueError('location must be specified')
    if(timestamp is None):
        if(timestamp_resolution is None):
            timestamp_resolution = 's'
        timestamp = timestamp_tools.create_timestamp_signal(timestamp_resolution,signal.size,0,1)
                       
    else:
        if(timestamp_resolution is None):
            raise ValueError('timestamp_resolution must be specified if timestamp is specified')
    
    fig.append_trace(go.Scatter(x=timestamp, y=signal, name=signal_name),row=location[0],col=location[1])
    if(peaks is not None):
        fig.append_trace(go.Scatter(x=timestamp[peaks], y=signal[peaks], name='Peaks',mode='markers',marker_color='red'),row=location[0],col=location[1])
    fig.update_layout(title=plot_title,xaxis_title='Time ('+timestamp_resolution+')',yaxis_title='Amplitude')
    
