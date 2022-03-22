from scipy.fft import fft
import numpy as np
from scipy import signal


def get_freq_features(sig,prefix='signal'):

    sig_features={}
    sig_fft = fft(np.array(sig))
    freqs, psd = signal.welch(sig_fft, return_onesided=False)
    f1sc = psd[np.where(np.logical_and(freqs > 0.1, freqs < 0.2))]
    f2sc = psd[np.where(np.logical_and(freqs > 0.2, freqs < 0.3))]
    f3sc = psd[np.where(np.logical_and(freqs > 0.3, freqs < 0.4))]

    sig_features[prefix+"_f1sc"] = f1sc.mean()
    sig_features[prefix+"_f2sc"] = f2sc.mean()
    sig_features[prefix+"_f3sc"] = f3sc.mean()
    
    return sig_features