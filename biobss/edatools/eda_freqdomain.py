import numpy as np
from numpy.typing import ArrayLike
from scipy import signal
from scipy.fft import fft

# Frequency-domain features
FREQ_FEATURES = {
    "f1sc": lambda x, freqs: x[np.where(np.logical_and(freqs > 0.1, freqs < 0.2))].mean(),
    "f2sc": lambda x, freqs: x[np.where(np.logical_and(freqs > 0.2, freqs < 0.3))].mean(),
    "f3sc": lambda x, freqs: x[np.where(np.logical_and(freqs > 0.3, freqs < 0.4))].mean(),
    "Energy": lambda x, freqs: np.sum(x ** 2),
    "Entropy": lambda x, freqs: np.sum(x * np.log(x)),
    "max_freq": lambda x, freqs: freqs[np.argmax(x)],
}


def eda_freq_features(sig: ArrayLike, prefix: str = "eda") -> dict:
    """Calculates frequency-domain EDA features.

    f1sc: Spectral power in the range of 0.1 to 0.2 Hz.
    f2sc: Spectral power in the range of 0.2 to 0.3 Hz.
    f3sc: Spectral power in the range of 0.3 to 0.4 Hz.
    Energy: Sum of the signal power
    Entropy: S sum of the power in the signal times the log of the power in the signal
    max_freq: Frequency corresponding to highest power in the signal

    Reference: Zangróniz, R., Martínez-Rodrigo, A., Pastor, J.M., López, M.T. and Fernández-Caballero, A., 2017.
    Electrodermal activity sensor for classification of calm/distress condition. Sensors, 17(10), p.2324.

    Args:
        sig (ArrayLike): EDA signal.
        prefix (str, optional): Prefix for the feature. Defaults to "eda".

    Returns:
        dict: Dictionary of calculated features.
    """
    sig_features = {}
    sig_fft = fft(np.array(sig))
    freqs, psd = signal.welch(sig_fft, return_onesided=False)

    for k, f in FREQ_FEATURES.items():
        try:
            sig_features["_".join([prefix, k])] = f(psd, freqs)
        except:
            sig_features["_".join([prefix, k])] = np.nan

    return sig_features
