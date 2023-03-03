import numpy as np
from numpy.typing import ArrayLike

def differentiate_ppg(sig: ArrayLike, sampling_rate:float) -> dict:
    """Calculates first and second derivatives of the PPG signal

    Args:
        sig (ArrayLike): PPG signal.
        sampling_rate (float): Sampling rate of the PPG signal (Hz).

    Returns:
        dict: VPG signal (first derivative), APG signal (second derivative)
    """

    vpg_sig = np.gradient(sig) / (1/sampling_rate)
    apg_sig = np.gradient(vpg_sig) / (1/sampling_rate)

    signals = {'vpg_sig':vpg_sig, 'apg_sig':apg_sig}
    print(signals)
    return signals