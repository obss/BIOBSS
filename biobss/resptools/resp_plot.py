import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

from ..plottools import *
from ..timetools import *


def plot_resp(
    signals: dict,
    peaks: dict = None,
    sampling_rate: float = None,
    timestamp_resolution: str = None,
    method: str = "matplotlib",
    show_peaks: bool = True,
    figsize: tuple = (18.5, 10.5),
    width: float = 1050,
    height: float = 600,
):
    """Generates plots for respiration signal(s).

    Args:
        signals (dict): The dictionary of signals to be plotted.
        peaks (dict, optional): The dictionary of peaks to be plotted. Defaults to None.
        sampling_rate (float, optional): Sampling rate of the signal. Defaults to None.
        timestamp_resolution (str, optional): Timestamp resolution. Defaults to None.
        method (str, optional): Package to generate plots. Defaults to 'matplotlib'.
        show_peaks (bool, optional): If True, peaks are plotted. Defaults to True.
        figsize (tuple, optional): Figure size for matplotlib. Defaults to (18.5, 10.5).
        width float, optional): Figure width for Plotly. Defaults to 1050.
        height (float, optional): Figure height for Plotly. Defaults to 600.

    Raises:
        ValueError: If method is not 'matplotlib' or 'plotly'.
    """
    if sampling_rate is not None:
        if timestamp_resolution is None:
            timestamp_resolution = "s"
        x_label = "Time (" + timestamp_resolution + ")"
    else:
        x_label = "Sample"

    if peaks is None:
        if show_peaks:
            raise ValueError("Peaks must be specified if show_peaks is True.")
        else:
            peaks = {}

    if method == "matplotlib":
        _plot_resp_matplotlib(
            signals=signals,
            peaks=peaks,
            sampling_rate=sampling_rate,
            timestamp_resolution=timestamp_resolution,
            x_label=x_label,
            figsize=figsize,
            show_peaks=show_peaks,
        )
    elif method == "plotly":
        _plot_resp_plotly(
            signals=signals,
            peaks=peaks,
            sampling_rate=sampling_rate,
            timestamp_resolution=timestamp_resolution,
            x_label=x_label,
            width=width,
            height=height,
            show_peaks=show_peaks,
        )
    else:
        raise ValueError("Undefined method.")


def _plot_resp_matplotlib(
    signals: dict,
    peaks: dict = None,
    sampling_rate: float = None,
    timestamp_resolution: str = "s",
    x_label=None,
    figsize: tuple = (18.5, 10.5),
    show_peaks: bool = True,
):
    """Generates plots for respiration signal(s) using Matplotlib."""
    # Create figure
    dim = len(signals)
    fig, axs = plt.subplots(dim, figsize=figsize)

    if dim == 1:
        axs = np.array([axs])

    # Plot respiratory signals
    i = 0
    for signal_name, signal in signals.items():
        y_values = signal["y"]

        if sampling_rate is not None:
            x_values = create_timestamp_signal(
                resolution=timestamp_resolution,
                length=len(y_values),
                rate=sampling_rate,
                start=y_values[0] / sampling_rate,
            )
            # x_label = 'Time (' + timestamp_resolution +')'
        else:
            x_values = np.linspace(0, len(y_values), len(y_values))
            # x_label = 'Sample'

        if signal_name not in peaks.keys():
            peaks[signal_name] = {}

        create_signal_plot_matplotlib(
            ax=axs[i],
            signal=y_values,
            x_values=x_values,
            show_peaks=show_peaks,
            peaks=peaks[signal_name],
            plot_title=signal_name,
            signal_name=signal_name,
            x_label=x_label,
        )

        i += 1

    fig.supxlabel(x_label)
    fig.supylabel("Amplitude")
    fig.tight_layout()
    plt.show()


def _plot_resp_plotly(
    signals: dict,
    peaks: dict = None,
    sampling_rate: float = None,
    timestamp_resolution="s",
    x_label=None,
    width: float = 1050,
    height: float = 600,
    show_peaks: bool = True,
):
    """Generates plots for respiration signal(s) using Plotly."""
    # Create figure
    dim = len(signals)
    fig = make_subplots(rows=dim, cols=1, shared_xaxes=True, x_title=x_label, y_title="Amplitude")

    i = 1
    for signal_name, signal in signals.items():
        y_values = signal["y"]

        if sampling_rate is not None:
            x_values = create_timestamp_signal(
                resolution=timestamp_resolution,
                length=len(y_values),
                rate=sampling_rate,
                start=y_values[0] / sampling_rate,
            )
            x_label = "Time (" + timestamp_resolution + ")"
        else:
            x_values = np.linspace(0, len(y_values), len(y_values))
            x_label = "Sample"

        if signal_name not in peaks.keys():
            peaks[signal_name] = {}

        create_signal_plot_plotly(
            fig,
            signal=y_values,
            x_values=x_values,
            show_peaks=show_peaks,
            peaks=peaks[signal_name],
            plot_title=signal_name,
            signal_name=signal_name,
            width=width,
            height=height,
            x_label=x_label,
            location=(i, 1),
        )
        i += 1

    fig.update_layout({"title": {"text": "Respiration Signal(s)", "x": 0.45, "y": 0.9}})
    fig.show()
