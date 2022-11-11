import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

from ..timetools import *
from ..plottools import *

def plot_ecg(signals:dict, peaks:dict=None, sampling_rate:float=None, timestamps=None, timestamp_resolution=None, method:str='matplotlib', show_peaks=True, figsize=(18.5, 10.5), width=800, height=440):

    ecg_raw=signals.get('Raw')

    if (timestamps is not None):
        if (len(timestamps) != len(ecg_raw)):
            raise ValueError('Timestamps and ECG signal must have the same length!')        

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

            x_values=create_timestamp_signal(resolution=timestamp_resolution, length=len(ecg_raw), rate=sampling_rate, start=0)
            x_label = 'Time (' + timestamp_resolution +')'

        else:
            x_values = np.linspace(0, len(ecg_raw), len(ecg_raw))
            x_label = 'Sample'

    if method == 'matplotlib':
        plot_ecg_matplotlib(signals=signals, peaks=peaks, x_values=x_values, x_label=x_label, figsize=figsize, show_peaks=show_peaks)

    elif method == 'plotly':
        plot_ecg_plotly(signals=signals, peaks=peaks, x_values=x_values, x_label=x_label, width=width, height=height, show_peaks=show_peaks)
    else:
        raise ValueError("Undefined method.")

def plot_ecg_matplotlib(signals:dict, peaks:dict=None, x_values:ArrayLike=None, x_label:str='Sample', figsize=(18.5, 10.5), show_peaks=True):
    
    # Create figure
    fig, axs = plt.subplots(figsize=figsize)
    #fig.set_size_inches(*figsize)

    #Plot raw ECG, filtered ECG and peaks
    for signal_name, signal in signals.items():
        create_signal_plot_matplotlib(ax=axs, signal=signal, x_values=x_values, plot_title=' ', signal_name=signal_name + ' ECG', x_label=x_label)
    
    if show_peaks:
        if peaks is not None:
            plot_peaks_matplotlib(axs, peaks=peaks, x_values=x_values, signal_name='ECG')
        else:
            raise ValueError("Peaks must be specified if show_peaks is True.")

    fig.supxlabel(x_label)
    fig.supylabel('Amplitude')
    plt.title('ECG Signal') 

    fig.tight_layout()
    plt.show()

def plot_ecg_plotly(signals:dict, peaks:dict=None, x_values:ArrayLike=None, x_label:str='Sample', width=800, height=440, show_peaks=True):

    # Create figure    
    fig = make_subplots(rows=1, cols=1)

    #Plot raw ECG, filtered ECG and peaks
    for signal_name, signal in signals.items():
        create_signal_plot_plotly(fig, signal, x_values=x_values, plot_title=' ',signal_name=signal_name + ' ECG', width=width, height=height, location=(1,1))

    if show_peaks:
        if peaks is not None:
            plot_peaks_plotly(fig, peaks=peaks, x_values=x_values, x_label=x_label, signal_name='ECG', location=(1,1))
        else:
            raise ValueError("Peaks must be specified if show_peaks is True.")

    fig.update_layout({'title': {'text': 'ECG Signal', 'x':0.45, 'y': 0.9}},xaxis_title=x_label,yaxis_title='Amplitude')
    fig.show()