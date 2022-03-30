import neurokit2 as nk
import numpy as np


def find_peaks(phasic_signal:np.ndarray, sampling_rate:float):

    # This function is a placeholder peak detection
    peak_signal, info = nk.eda_peaks(
        phasic_signal.values,
        sampling_rate=sampling_rate,
        method="neurokit",
        amplitude_min=0.1,
    )

    return peak_signal, info
