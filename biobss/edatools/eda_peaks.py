from typing import Tuple

import neurokit2 as nk
import pandas as pd
from numpy.typing import ArrayLike


def eda_detectpeaks(phasic_signal: ArrayLike, sampling_rate: float) -> Tuple:
    """Detects peaks from phasic component of EDA signal.

    Args:
        phasic_signal (ArrayLike): Phasic EDA signal.
        sampling_rate (float): Sampling rate of the EDA signal (Hz).

    Raises:
        ValueError: If sampling rate is not greater than 0.

    Returns:
        Tuple: Peak array, info
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    if not isinstance(phasic_signal, pd.Series):
        phasic_signal = pd.Series(phasic_signal)
    # This function is a placeholder peak detection
    peak_signal, info = nk.eda_peaks(
        phasic_signal.values,
        sampling_rate=sampling_rate,
        method="neurokit",
        amplitude_min=0.1,
    )

    return peak_signal, info
