from scipy.fft import fft
import numpy as np
from scipy import signal
from numpy.typing import ArrayLike

def get_freq_features(sig:ArrayLike, prefix="signal")->dict:
    """This method calculates Frequency features over the given signal.
    
    In regards to frequency aspects, the s'gnal is transformed into the frequency domain by
    using a nonparametric FFT algorithm. Then, the spectral power in bandwidths 0.1 to 0.2 (F1SC), 0.2 to
    0.3 (F2SC) and 0.3 to 0.4 (F3SC) Hz are estimated.
    
    Zangróniz, R., Martínez-Rodrigo, A., Pastor, J.M., López, M.T. and Fernández-Caballero, A., 2017. 
    Electrodermal activity sensor for classification of calm/distress condition. Sensors, 17(10), p.2324.
    
    Args:
        sig (ArrayLike): 1-D signal
        prefix (str, optional): prefix for signal name. Defaults to "signal".

    Returns:
        dict: Frequency features
    """

    sig_features = {}
    sig_fft = fft(np.array(sig))
    freqs, psd = signal.welch(sig_fft, return_onesided=False)
    f1sc = psd[np.where(np.logical_and(freqs > 0.1, freqs < 0.2))]
    f2sc = psd[np.where(np.logical_and(freqs > 0.2, freqs < 0.3))]
    f3sc = psd[np.where(np.logical_and(freqs > 0.3, freqs < 0.4))]

    sig_features[prefix + "_f1sc"] = f1sc.mean()
    sig_features[prefix + "_f2sc"] = f2sc.mean()
    sig_features[prefix + "_f3sc"] = f3sc.mean()

    return sig_features
