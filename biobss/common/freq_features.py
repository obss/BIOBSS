from scipy.fft import fft
import numpy as np
from scipy import signal
from numpy.typing import ArrayLike

# Frequency Features 

FREQ_FEATURES={
    "f1sc": lambda x,freqs: x[np.where(np.logical_and(freqs > 0.1, freqs < 0.2))].mean(),
    "f2sc": lambda x,freqs: x[np.where(np.logical_and(freqs > 0.2, freqs < 0.3))].mean(),
    "f3sc": lambda x,freqs: x[np.where(np.logical_and(freqs > 0.3, freqs < 0.4))].mean(),
    "Energy": lambda x: np.sum(x**2),
    "Entropy": lambda x: np.sum(x*np.log(x)),
    "max_freq": lambda x,freqs: freqs[np.argmax(x)],
}

def get_freq_features(sig:ArrayLike, prefix="signal")->dict:
    """This method calculates Frequency features over the given signal.
    
    In regards to frequency aspects, the signal is transformed into the frequency domain by
    using a nonparametric FFT algorithm. Then, the spectral power in bandwidths 0.1 to 0.2 (F1SC), 0.2 to
    0.3 (F2SC) and 0.3 to 0.4 (F3SC) Hz are estimated.
    
    Entropy and Energy are also calculated.
    Energy : sum of the power in the signal
    Entropy : The entropy of the signal is the sum of the power in the signal times the log of the power in the signal.
    Max Frequency : The frequency with the highest power in the signal.
    
    
    Zangróniz, R., Martínez-Rodrigo, A., Pastor, J.M., López, M.T. and Fernández-Caballero, A., 2017. 
    Electrodermal activity sensor for classification of calm/distress condition. Sensors, 17(10), p.2324.
    
    Args:
        sig (ArrayLike): 1-D signal
        prefix (str, optional): prefix for the signal name. Defaults to "signal".

    Returns:
        dict: Frequency features
    """

    sig_features = {}
    sig_fft = fft(np.array(sig))
    freqs, psd = signal.welch(sig_fft, return_onesided=False)

    for k, f in FREQ_FEATURES.items():
        sig_features["_".join([prefix, k])] = f(psd, freqs)

    return sig_features
