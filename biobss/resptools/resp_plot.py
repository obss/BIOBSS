import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

from ..timetools import *
from ..plottools import *

def plot_resp(signals:dict, peaks:dict=None, sampling_rate:float=None, timestamp_resolution=None, method:str='matplotlib', show_peaks=True, figsize=(18.5, 10.5), width=1050, height=600):

    if sampling_rate is not None:
        if (timestamp_resolution is None):
            timestamp_resolution='s'
        x_label = 'Time (' + timestamp_resolution +')'
    else:
        x_label = 'Sample'

    if method == 'matplotlib':
        plot_resp_matplotlib(signals=signals, peaks=peaks, sampling_rate=sampling_rate, timestamp_resolution=timestamp_resolution, x_label=x_label, figsize=figsize, show_peaks=show_peaks)
    elif method == 'plotly':
        plot_resp_plotly(signals=signals, peaks=peaks, sampling_rate=sampling_rate, timestamp_resolution=timestamp_resolution, x_label=x_label, width=width, height=height, show_peaks=show_peaks)
    else:
        raise ValueError("Undefined method.")

def plot_resp_matplotlib(signals:dict, peaks:dict=None, sampling_rate:float=None, timestamp_resolution:str='s', x_label= None, figsize=(18.5, 10.5), show_peaks=True):

    # Create figure
    dim = len(signals)
    fig, axs = plt.subplots(dim, figsize=figsize)
    #fig.set_size_inches(*figsize)

    if dim == 1:
        axs = np.array([axs])

    #Plot respiratory signals
    i=0
    for signal_name, signal in signals.items():
        y_values = signal['y']

        if sampling_rate is not None:
            x_values=create_timestamp_signal(resolution=timestamp_resolution, length=len(y_values), rate=sampling_rate, start=y_values[0]/sampling_rate)
            #x_label = 'Time (' + timestamp_resolution +')'
        else:
            x_values = np.linspace(0, len(y_values), len(y_values))
            #x_label = 'Sample'
            
        create_signal_plot_matplotlib(ax=axs[i], signal=y_values, x_values=x_values, plot_title=signal_name, signal_name=signal_name, x_label=x_label)

        if show_peaks:
            if peaks[signal_name] is not None:
                plot_peaks_matplotlib(axs[i], peaks=peaks, x_values=x_values)
            else:
                raise ValueError("Peaks must be specified if show_peaks is True.")
        
        i += 1

    fig.supxlabel(x_label)
    fig.supylabel('Amplitude')

    fig.tight_layout()
    plt.show()        

def plot_resp_plotly(signals:dict, peaks:dict=None, sampling_rate:float=None, timestamp_resolution='s', x_label=None, width= 1050, height=600, show_peaks=True):
 
    # Create figure
    dim = len(signals)
    fig = make_subplots(rows=dim, cols=1, shared_xaxes=True, x_title=x_label, y_title='Amplitude')

    i=1
    for signal_name, signal in signals.items():
        y_values = signal['y']

        if sampling_rate is not None:
            x_values=create_timestamp_signal(resolution=timestamp_resolution, length=len(y_values), rate=sampling_rate, start=y_values[0]/sampling_rate)
            x_label = 'Time (' + timestamp_resolution +')'
        else:
            x_values = np.linspace(0, len(y_values), len(y_values))
            x_label = 'Sample'
            
        create_signal_plot_plotly(fig, signal=y_values, x_values=x_values, plot_title=signal_name,signal_name=signal_name, x_label=x_label, location=(i,1))

        if show_peaks:
            if peaks[signal_name] is not None:
                plot_peaks_plotly(fig, peaks=peaks, x_values=x_values, x_label=x_label)    
            else:
                raise ValueError("Peaks must be specified if show_peaks is True.")

        i += 1

    fig.update_layout({'title': {'text': 'Respiration Signal(s)', 'x':0.45, 'y': 0.9}})
    
    fig.show()