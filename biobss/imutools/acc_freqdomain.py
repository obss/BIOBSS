import numpy as np
from scipy import stats, signal, fft
from numpy.typing import ArrayLike

from biobss.common.signal_power import *
from biobss.common.signal_fft import *
from biobss.common.signal_psd import *

FREQ_FEATURES = {
    "fft_mean": lambda sigfft, _0, _1, _2: np.mean(sigfft), 
    "fft_std": lambda sigfft, _0, _1, _2: np.std(sigfft),
    "fft_mad": lambda sigfft, _0, _1, _2: np.mean(np.abs(sigfft - np.mean(sigfft))),
    "fft_min": lambda sigfft, _0, _1, _2: np.min(sigfft),
    "fft_max": lambda sigfft, _0, _1, _2: np.max(sigfft),
    "fft_range": lambda sigfft, _0, _1, _2: np.max(sigfft) - np.min(sigfft),
    "fft_median": lambda sigfft, _0, _1, _2: np.median(sigfft),
    "fft_medad": lambda sigfft, _0, _1, _2: np.median(np.abs(sigfft - np.median(sigfft))),
    "fft_iqr": lambda sigfft, _0, _1, _2: np.percentile(sigfft, 75) - np.percentile(sigfft, 25),
    "fft_abmean": lambda sigfft, _0, _1, _2: np.sum(sigfft > np.mean(sigfft)),
    "fft_npeaks": lambda sigfft, _0, _1, _2: len(signal.find_peaks(sigfft)[0]),
    "fft_skew": lambda sigfft, _0, _1, _2: stats.skew(sigfft),
    "fft_kurtosis": lambda sigfft, _0, _1, _2: stats.kurtosis(sigfft),
    "fft_energy": lambda sigfft, _0, _1, _2: np.sum(sigfft**2)/100,
    "fft_entropy": lambda sigfft, _0, _1, _2: np.sum(sigfft*np.log(sigfft)),
    "f1sc": lambda _0,_1,pxx,fxx: sig_power(pxx,fxx,[0.1,0.2]),
    "f2sc": lambda _0,_1,pxx,fxx: sig_power(pxx,fxx,[0.2,0.3]),
    "f3sc": lambda _0,_1,pxx,fxx: sig_power(pxx,fxx,[0.3,0.4]),
    "max_freq": lambda sigfft, freq, _0, _1: fft_peaks(sigfft,freq,1,loc=True),
}


def get_freq_features(sig: ArrayLike, sampling_rate, prefix) -> dict:
    """Calculates frequency-domain features.

    From https://towardsdatascience.com/feature-engineering-on-time-series-data-transforming-signal-data-of-a-smartphone-accelerometer-for-72cbe34b8a60

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
        sig (ArrayLike): Input signal
        sampling_rate (_type_): Sampling rate
        prefix (_type_): Prefix 

    Returns:
        dict: Dictionary of statistical features
    """

    features_freq={}

    freq, sigfft = sig_fft(sig=sig, sampling_rate=sampling_rate)

    #nfft=len(sig)
    #freq=fft.fftfreq(nfft,1/sampling_rate)
    #sigfft=np.abs(fft.fft(sig,nfft)/len(sig))
    #P1=sigfft[0:int(nfft/2)]
    #P1[1:-1] = 2 * P1[1:-1]
    #sigfft=P1
    #freq=freq[0 : int(len(sig)/ 2)]

    f, pxx = sig_psd(sig=sig, sampling_rate=sampling_rate, method='welch')

    #nfft=len(sig)
    #sig_=sig-np.mean(sig)
    #f,pxx=signal.welch(sig_, fs=sampling_rate, nfft=nfft)
    
    features_freq={}
    for key,func in FREQ_FEATURES.items():
        features_freq["_".join([prefix, key])]=func(sigfft,freq,pxx,f)

    return features_freq

        
         
