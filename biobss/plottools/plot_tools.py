import matplotlib.pyplot as plt
import numpy as np
from ..timetools import *
from numpy.typing import ArrayLike
import plotly.graph_objects as go
from plotly_resampler import register_plotly_resampler

<<<<<<< HEAD

def create_signal_plot(signal,ax,timestamp=None,timestamp_resolution=None,plot_title="Signal Plot",signal_name="Signal",peaks=None):
    
    if(timestamp is None):
        if(timestamp_resolution is None):
            timestamp_resolution = 's'
        timestamp = timestamp_tools.create_timestamp_signal(timestamp_resolution,signal.size,0,1)
                       
    else:
        if(timestamp_resolution is None):
            raise ValueError('timestamp_resolution must be specified if timestamp is specified')
    

=======
def create_signal_plot_matplotlib(ax, signal:ArrayLike=None, x_values=None, plot_title="Signal Plot", signal_name="Signal", x_label='Sample'):

    if (x_values is None):
        x_values = np.linspace(0, len(signal), len(signal))
                          
>>>>>>> a5c1edbc44fa916c296ebb316ff9414c2bb6d232
    # Check if there is existing legend
    if(isinstance(ax.get_legend(),type(None))):
        legend = []
    else:
        legend = [x.get_text() for x in ax.get_legend().texts]

<<<<<<< HEAD
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
    
=======
    ax.plot(x_values,signal)
    ax.set_title(plot_title)
    ax.set_xlim([0, max(x_values)])
    #ax.set_xlabel(x_label)
    #ax.set_ylabel('Amplitude')
    legend.append(signal_name)

    ax.legend(legend, loc='center left', bbox_to_anchor=(1.0, 0.5))
    
def plot_peaks_matplotlib(ax, peaks:dict=None, x_values=None, x_label='Sample', signal_name='Signal'):
    legend = [x.get_text() for x in ax.get_legend().texts]
    for peak_type, peak_dict in peaks.items():
        peak_loc = peak_dict['Location']
        peak_amp = peak_dict['Amplitude']
        ax.scatter(x_values[peak_loc], peak_amp, label=peak_type)
        legend.append(signal_name+ ' ' +peak_type)
    ax.legend(legend, loc='center left', bbox_to_anchor=(1.0, 0.5))

def create_signal_plot_plotly(fig, signal:ArrayLike=None,x_values=None,plot_title="Signal Plot",signal_name="Signal", x_label='Sample', width=1050, height=600, location=None):
    
    if (x_values is None):
        x_values = np.linspace(0, len(signal), len(signal))

    register_plotly_resampler(mode='auto')

    if(location is None):
        raise ValueError('location must be specified')

    fig.append_trace(go.Scatter(x=x_values, y=signal, name=signal_name), row=location[0], col=location[1])
    
    fig.update_layout({'xaxis': {'range':[0, x_values.max()]}}, title=plot_title, width=width, height=height)
    fig.add_annotation(xref="x domain", yref="y domain", x=0.5, y=1.2, showarrow=False, text=plot_title, row=location[0], col=location[1])

def plot_peaks_plotly(fig, peaks:dict=None, x_values=None, x_label='Sample', signal_name="Signal", location=None):

    if(location is None):
        raise ValueError('location must be specified')

    for peak_type, peak_dict in peaks.items():
        peak_loc = peak_dict['Location']
        peak_amp = peak_dict['Amplitude']
        fig.append_trace(go.Scatter(x=x_values[peak_loc], y=peak_amp, name=signal_name+ ' ' +peak_type, mode='markers'),row=location[0],col=location[1])
    

>>>>>>> a5c1edbc44fa916c296ebb316ff9414c2bb6d232
