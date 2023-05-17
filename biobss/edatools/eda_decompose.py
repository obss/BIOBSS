import cvxopt as cv
import neurokit2 as nk
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def eda_decompose(eda_signal: ArrayLike, sampling_rate: float, method: str = "highpass") -> pd.DataFrame:
    """Decomposes EDA signal into tonic and phasic components.

    Args:
        eda_signal (ArrayLike): EDA signal.
        sampling_rate (float): Sampling rate of EDA signal (Hz).
        method (str, optional): Method to be used for decomposition. Defaults to "highpass".

    Raises:
        ValueError: If sampling rate is not greater than 0.
        Exception: If method is not implemented.

    Returns:
        pd.DataFrame: A dataframe composed of Phasic and Tonic components of EDA signal
    """

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    method = method.lower()

    if method == "cvxeda":
        raise Exception("Method not implemented")

    elif method == "highpass":
        decomposed = _eda_highpass(eda_signal, sampling_rate)  # Default

    elif method == "bandpass":
        decomposed = _eda_bandpass(eda_signal, sampling_rate)
    else:
        raise Exception("Method not implemented")

    return decomposed


def _eda_highpass(eda_signal: ArrayLike, sampling_rate: float) -> pd.DataFrame:

    # Highpass filter for EDA signal decomposition
    phasic = nk.signal_filter(eda_signal, sampling_rate=sampling_rate, lowcut=0.05, method="butter")
    tonic = nk.signal_filter(eda_signal, sampling_rate=sampling_rate, highcut=0.05, method="butter")

    out = pd.DataFrame({"EDA_Tonic": np.array(tonic), "EDA_Phasic": np.array(phasic)})

    return out


def _eda_bandpass(eda_signal: ArrayLike, sampling_rate: float) -> pd.DataFrame:

    # Bandpass filter for EDA signal decomposition
    phasic = nk.signal_filter(eda_signal, sampling_rate, 0.2, 1)
    tonic = nk.signal_filter(eda_signal, sampling_rate, highcut=0.2)

    out = pd.DataFrame({"EDA_Tonic": np.array(tonic), "EDA_Phasic": np.array(phasic)})

    return out
