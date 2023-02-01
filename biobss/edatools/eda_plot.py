import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

from ..plottools import *
from ..timetools import *


def plot_eda(
    signals: dict,
    peaks: dict = None,
    sampling_rate: float = None,
    timestamps: ArrayLike = None,
    timestamp_resolution: str = None,
    method: str = "matplotlib",
    show_peaks: bool = True,
    figsize: tuple = (18.5, 10.5),
    width: float = 1050,
    height: float = 600,
):
    """Generates plots for EDA signal.

    Args:
        signals (dict): The dictionary of signals to be plotted.
        peaks (dict, optional): The dictionary of peaks to be plotted. Defaults to None.
        sampling_rate (float, optional): Sampling rate of the signal. Defaults to None.
        timestamps (ArrayLike, optional): Timestamp array. Defaults to None.
        timestamp_resolution (str, optional): Timestamp resolution. Defaults to None.
        method (str, optional): Package to generate plots. It can be 'matplotlib' or 'plotly'. Defaults to 'matplotlib'.
        show_peaks (bool, optional): If True, peaks are plotted. Defaults to True.
        figsize (tuple, optional): Figure size for matplotlib. Defaults to (18.5, 10.5) inches.
        width (float, optional): Figure width for Plotly. Defaults to 800 pixels.
        height (float, optional): Figure height for Plotly. Defaults to 440 pixels.

    Raises:
        ValueError: If timestamps is not None and timestamp resolution is not provided.
        ValueError: If timestamps array and EDA signal have different lengths.
        ValueError: If method is not 'matplotlib' or 'plotly'.
    """
    eda_raw = signals.get("Raw")

    if timestamps is not None:
        if len(timestamps) != len(eda_raw):
            raise ValueError("Timestamps and PPG signal must have the same length!")

        if timestamp_resolution is None:
            raise ValueError("Timestamp resolution must be provided if timestamps are provided!")
        else:
            timestamp_resolution = timestamp_resolution

        x_values = timestamps
        x_label = "Time (" + timestamp_resolution + ")"

    else:
        if sampling_rate is not None:
            if timestamp_resolution is None:
                timestamp_resolution = "s"

            x_values = create_timestamp_signal(
                resolution=timestamp_resolution, length=len(eda_raw), rate=sampling_rate, start=0
            )
            x_label = "Time (" + timestamp_resolution + ")"

        else:
            x_values = np.linspace(0, len(eda_raw), len(eda_raw))
            x_label = "Sample"

    if peaks is None:
        if show_peaks:
            raise ValueError("Peaks must be specified if show_peaks is True.")
        else:
            peaks = {}

    if method == "matplotlib":
        _plot_eda_matplotlib(
            signals=signals, peaks=peaks, x_values=x_values, x_label=x_label, figsize=figsize, show_peaks=show_peaks
        )

    elif method == "plotly":
        _plot_eda_plotly(
            signals=signals,
            peaks=peaks,
            x_values=x_values,
            x_label=x_label,
            width=width,
            height=height,
            show_peaks=show_peaks,
        )
    else:
        raise ValueError("Undefined method.")


def _plot_eda_matplotlib(
    signals: dict,
    peaks: dict = None,
    x_values: ArrayLike = None,
    x_label: str = "Sample",
    figsize=(18.5, 10.5),
    show_peaks=True,
):

    # Create figure
    dim = len(signals)
    fig, axs = plt.subplots(dim, figsize=figsize)
    # fig.set_size_inches(*figsize)

    if dim == 1:
        axs = np.array([axs])

    # Plot raw EDA, cleaned EDA, phasic EDA and tonic EDA
    i = 0
    for signal_name, signal in signals.items():

        if signal_name not in peaks.keys():
            peaks[signal_name] = {}

        create_signal_plot_matplotlib(
            ax=axs[i],
            signal=signal,
            x_values=x_values,
            show_peaks=show_peaks,
            peaks=peaks[signal_name],
            plot_title=" ",
            signal_name=signal_name + " EDA",
            x_label=x_label,
        )

        i += 1

    fig.supxlabel(x_label)
    fig.supylabel("Amplitude")
    plt.title("EDA Signal")

    fig.tight_layout()
    plt.show()


def _plot_eda_plotly(
    signals: dict,
    peaks: dict = None,
    x_values: ArrayLike = None,
    x_label: str = "Sample",
    width=1050,
    height=600,
    show_peaks=True,
):

    # Create figure
    dim = len(signals)
    fig = make_subplots(rows=dim, cols=1, shared_xaxes=True, x_title=x_label, y_title="Amplitude")

    # Plot raw EDA, cleaned EDA, phasic EDA and tonic EDA
    i = 1
    for signal_name, signal in signals.items():

        if signal_name not in peaks.keys():
            peaks[signal_name] = {}

        create_signal_plot_plotly(
            fig,
            signal=signal,
            x_values=x_values,
            show_peaks=show_peaks,
            peaks=peaks[signal_name],
            plot_title=" ",
            signal_name=signal_name + " EDA",
            x_label=x_label,
            width=width,
            height=height,
            location=(i, 1),
        )

        i += 1

    fig.update_layout({"title": {"text": "EDA Signals", "x": 0.45, "y": 0.9}})
    fig.show()
