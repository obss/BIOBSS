<<<<<<< HEAD
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

=======
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

from ..timetools import *
from ..plottools import *


def plot_eda(signals:dict, peaks:dict=None, sampling_rate:float=None, timestamps=None, timestamp_resolution=None, method:str='matplotlib', show_peaks=True, figsize=(18.5, 10.5), width=1050, height=600):
    
    eda_raw=signals.get('Raw')
   
    if (timestamps is not None):
        if (len(timestamps) != len(eda_raw)):
            raise ValueError('Timestamps and PPG signal must have the same length!')        

        if (timestamp_resolution is None):
            raise ValueError('Timestamp resolution must be provided if timestamps are provided!')        
        else:
            timestamp_resolution=timestamp_resolution        
            
        x_values = timestamps
        x_label = 'Time (' + timestamp_resolution +')'

    else:
        if sampling_rate is not None:
            if (timestamp_resolution is None):
                timestamp_resolution='s'

            x_values=create_timestamp_signal(resolution=timestamp_resolution, length=len(eda_raw), rate=sampling_rate, start=0)
            x_label = 'Time (' + timestamp_resolution +')'

        else:
            x_values = np.linspace(0, len(eda_raw), len(eda_raw))
            x_label = 'Sample'

    if method == 'matplotlib':
        plot_eda_matplotlib(signals=signals, peaks=peaks, x_values=x_values, x_label=x_label, figsize=figsize, show_peaks=show_peaks)

    elif method == 'plotly':
        plot_eda_plotly(signals=signals, peaks=peaks, x_values=x_values, x_label=x_label, width=width, height=height, show_peaks=show_peaks)
    else:
        raise ValueError("Undefined method.")

def plot_eda_matplotlib(signals:dict, peaks:dict=None, x_values:ArrayLike=None, x_label:str='Sample', figsize=(18.5, 10.5), show_peaks=True):

    # Create figure
    dim = len(signals)  
    fig, axs = plt.subplots(dim, figsize=figsize)
    #fig.set_size_inches(*figsize)

    if dim == 1:
        axs = np.array([axs])

    # Plot raw EDA, cleaned EDA, phasic EDA and tonic EDA
    i=0
    for signal_name, signal in signals.items():
        create_signal_plot_matplotlib(ax=axs[i], signal=signal, x_values=x_values, plot_title=' ', signal_name=signal_name + ' EDA', x_label=x_label)

        if show_peaks:
            if peaks is not None:
                if peaks[signal_name]:
                    plot_peaks_matplotlib(axs[i], peaks=peaks[signal_name], x_values=x_values, x_label=x_label, signal_name='EDA')
            else:
                raise ValueError("Peaks must be specified if show_peaks is True.")
    
        i += 1
    
    fig.supxlabel(x_label)
    fig.supylabel('Amplitude')

    fig.tight_layout()
    plt.show()


def plot_eda_plotly(signals:dict, peaks:dict=None, x_values:ArrayLike=None, x_label:str='Sample', width=1050, height=600, show_peaks=True):

    # Create figure 
    dim = len(signals)    
    fig = make_subplots(rows=dim, cols=1, shared_xaxes=True, x_title=x_label, y_title='Amplitude')

    # Plot raw EDA, cleaned EDA, phasic EDA and tonic EDA
    i=1
    for signal_name, signal in signals.items():
        create_signal_plot_plotly(fig, signal=signal, x_values=x_values, plot_title=signal_name, signal_name=+ ' ' + signal_name, x_label=x_label, width=width, height=height, location=(i,1))

        if show_peaks:
            if peaks is not None:
                if peaks[signal_name]:
                    plot_peaks_plotly(fig, peaks=peaks[signal_name], x_values=x_values, x_label=x_label, signal_name= signal_name, location=(i,1))
            else:
                raise ValueError("Peaks must be specified if show_peaks is True.")
    
        i += 1
 
    fig.update_layout({'title': {'text': 'EDA Signals', 'x':0.45, 'y': 0.9}})
    fig.show()
    
>>>>>>> a5c1edbc44fa916c296ebb316ff9414c2bb6d232
