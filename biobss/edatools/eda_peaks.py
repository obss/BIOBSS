import neurokit2 as nk
import numpy as np


def eda_detectpeaks(phasic_signal: np.ndarray, sampling_rate: float):
    
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    # This function is a placeholder peak detection
    peak_signal, info = nk.eda_peaks(
        phasic_signal.values,
        sampling_rate=sampling_rate,
        method="neurokit",
        amplitude_min=0.1,
    )

    return peak_signal, info
