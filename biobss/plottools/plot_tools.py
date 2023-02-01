import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from plotly_resampler import register_plotly_resampler

from ..timetools import *


def create_signal_plot_matplotlib(
    ax: plt.Axes,
    signal: ArrayLike = None,
    x_values=None,
    show_peaks: bool = False,
    peaks: dict = None,
    plot_title: str = "Signal Plot",
    signal_name: str = "Signal",
    x_label: str = "Sample",
):
    """Generates plots for given signals using Matplotlib.

    Args:
        ax (plt.Axes): Axes to plot signal.
        signal (ArrayLike, optional): Array of y-axis values. Defaults to None.
        x_values (_type_, optional): Array of x-axis values. Defaults to None.
        show_peaks (bool, optional): If True, peaks are plotted. Defaults to False.
        peaks (dict, optional): Dictionary of peaks to be plotted. Defaults to None.
        plot_title (str, optional): Plot title. Defaults to "Signal Plot".
        signal_name (str, optional): Name of signal to be plotted. Defaults to "Signal".
        x_label (str, optional): Label of x-axis. Defaults to 'Sample'.
    """
    if x_values is None:
        x_values = np.linspace(0, len(signal), len(signal))

    # Check if there is existing legend
    if isinstance(ax.get_legend(), type(None)):
        legend = []
    else:
        legend = [x.get_text() for x in ax.get_legend().texts]

    ax.plot(x_values, signal)
    legend.append(signal_name)

    if show_peaks:
        for peak_type, peak_loc in peaks.items():
            peak_amp = signal[peak_loc]
            ax.scatter(x_values[peak_loc], peak_amp, label=peak_type)
            legend.append(signal_name + " " + peak_type)

    ax.set_title(plot_title)
    ax.set_xlim([0, max(x_values)])
    # ax.set_xlabel(x_label)
    # ax.set_ylabel('Amplitude')

    ax.legend(legend, loc="center left", bbox_to_anchor=(1.0, 0.5))


def create_signal_plot_plotly(
    fig: go.Figure,
    signal: ArrayLike = None,
    x_values: ArrayLike = None,
    show_peaks: bool = False,
    peaks: dict = None,
    plot_title: str = "Signal Plot",
    signal_name: str = "Signal",
    x_label: str = "Sample",
    width: float = 1050,
    height: float = 600,
    location: tuple = None,
):
    """Generates plots for given signals using Plotly.

    Args:
        fig (go.Figure): Figure to plot signal.
        signal (ArrayLike, optional): Array of y-axis values. Defaults to None.
        x_values (ArrayLike, optional): Array of x-axis values. Defaults to None.
        show_peaks (bool, optional): If True, peaks are plotted. Defaults to False.
        peaks (dict, optional): Dictionary of peaks to be plotted. Defaults to None.
        plot_title (str, optional): Plot title. Defaults to "Signal Plot".
        signal_name (str, optional): Name of signal to be plotted. Defaults to "Signal".
        x_label (str, optional): Label of x-axis. Defaults to 'Sample'.
        width (float, optional): Figure width. Defaults to 1050.
        height (float, optional): Figure height. Defaults to 600.
        location (tuple, optional): Subplot location. Defaults to None.

    Raises:
        ValueError: If location is not provided.
    """
    # adjust it
    limit = 200000

    if len(signal) > limit:
        Warning("Signal is too large and will be resampled. Consider using create_signal_plot instead")
        register_plotly_resampler(mode="auto")

    if x_values is None:
        x_values = np.linspace(0, len(signal), len(signal))

    if location is None:
        raise ValueError("Location must be specified")

    fig.append_trace(go.Scatter(x=x_values, y=signal, name=signal_name), row=location[0], col=location[1])

    if show_peaks:

        for peak_type, peak_loc in peaks.items():
            peak_amp = signal[peak_loc]
            fig.append_trace(
                go.Scatter(x=x_values[peak_loc], y=peak_amp, name=signal_name + " " + peak_type, mode="markers"),
                row=location[0],
                col=location[1],
            )

    fig.update_layout({"xaxis": {"range": [0, x_values.max()]}}, title=plot_title, width=width, height=height)
    fig.add_annotation(
        xref="x domain",
        yref="y domain",
        x=0.5,
        y=1.2,
        showarrow=False,
        text=plot_title,
        row=location[0],
        col=location[1],
    )
