import numpy as np
from scipy import stats, signal
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


def get_freq_features(signals: ArrayLike, signal_names:ArrayLike, sampling_rate, prefix='acc') -> dict:
    """Calculates frequency-domain features.

    From 
        https://towardsdatascience.com/feature-engineering-on-time-series-data-transforming-signal-data-of-a-smartphone-accelerometer-for-72cbe34b8a60
        Zangróniz, R., Martínez-Rodrigo, A., Pastor, J.M., López, M.T. and Fernández-Caballero, A., 2017. 
        Electrodermal activity sensor for classification of calm/distress condition. Sensors, 17(10), p.2324.

    fft_mean': mean of fft peaks
    fft_std': standard deviation of fft peaks
    fft_mad': mean absolute deviation of fft peaks
    fft_min': minimum value of fft peaks
    fft_max': maximum value of fft peaks
    fft_range': difference of maximum and minimum values of fft peaks
    fft_median': median value of fft peaks
    fft_medad': median absolute deviation of fft peaks
    fft_iqr': interquartile range of fft peaks
    fft_abmean': number of fft peaks above mean
    fft_npeaks': number of fft peaks
    fft_skew': skewness of fft peaks
    fft_kurtosis': kurtosis of fft peaks
    fft_energy': energy of fft peaks
    fft_entropy': entropy of fft peaks
    f1sc': signal power in the range of 0.1 to 0.2 Hz
    f2sc': signal power in the range of 0.2 to 0.3 Hz
    f3sc': signal power in the range of 0.3 to 0.4 Hz
    max_freq': frequency of maximum fft peak

    Args:
        sig (ArrayLike): Signal to be analyzed.
        sampling_rate (float): Sampling rate of the signal (Hz).
        prefix (str): Prefix. Defaults to None.

    Returns:
        dict: Dictionary of statistical features
    """
    data = dict(zip(signal_names, signals))

    features_freq={}
    
    for signal_name, signal in data.items():

        freq, sigfft = sig_fft(sig=signal, sampling_rate=sampling_rate)
        f, pxx = sig_psd(sig=signal, sampling_rate=sampling_rate, method='welch')
    
        features_freq={}

        for key,func in FREQ_FEATURES.items():
            features_freq["_".join([signal_name, key])]=func(sigfft,freq,pxx,f)

    return features_freq

        
         
