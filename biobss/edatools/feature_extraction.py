import neurokit2 as nk
import numpy as np
import pandas as pd
from matplotlib import docstring
from scipy import signal, stats
from scipy.fft import fft
from .hjorth import *
from .eda_decompose import *
from .signal_features import *
import multiprocessing
from functools import partial



def from_decomposed(signal_phasic, signal_tonic, sr):

    phasic_features = from_scr(signal_phasic, sr)
    tonic_features = from_scl(signal_tonic, sr)
    
    return phasic_features, tonic_features


def from_signal(signal, sr=20):

    decomposed_ = eda_decompose(signal, sr)
    eda_phasic = decomposed_['EDA_Phasic']
    eda_tonic = decomposed_['EDA_Tonic']
    phasic_features,tonic_features = from_decomposed(eda_phasic, eda_tonic, sr)
    return phasic_features,tonic_features

def from_windows(eda_windows,sr=20,parallel=False,n_jobs=6):
    
    if(parallel):
        f_pool=multiprocessing.Pool(processes=n_jobs)
        features=f_pool.map(partial(from_signal,sr=sr),eda_windows)
    else:
        features=[]
        for w in eda_windows:
            features.append(from_signal(w,sr))
        features=pd.DataFrame(features)
    return features

def from_decomposed_windows(phasic_windows, tonic_windows, sr,parallel=False,n_jobs=6):

    scr_features = []
    scl_features = []
    if(parallel):
        f_pool=multiprocessing.Pool(processes=n_jobs)
        scr_features=f_pool.map(partial(from_scr,sr=sr),phasic_windows)
        f_pool=multiprocessing.Pool(processes=n_jobs)
        scl_features=f_pool.map(partial(from_scl,sr=sr),tonic_windows)
    
    else:    
        for scrw in phasic_windows:
            scr_features.append(from_scr(scrw, sr))

        for sclw in tonic_windows:
            scl_features.append(from_scl(sclw, sr))

    scr_features = pd.DataFrame(scr_features)
    scl_features = pd.DataFrame(scl_features)
    return pd.concat([scr_features, scl_features],axis=0, ignore_index=True)


def from_scr(signal_phasic, sr):

    features = {}
    min = np.min(signal_phasic)
    max = np.max(signal_phasic)
    features['scr_mean'] = signal_phasic.mean()
    features['scr_std'] = np.std(signal_phasic)
    features['scr_max'] = max
    features['scr_min'] = max
    features["scr_dynamic_range"] = (max-min)
    s_d1 = np.gradient(signal_phasic)
    features["fmsc"] = s_d1.mean()
    features["fdsc"] = np.std(s_d1)
    s_d2 = np.gradient(s_d1)
    features["smsc"] = s_d2.mean()
    features["sdsc"] = np.std(s_d2)
    alsc = calculate_alsc(signal_phasic)
    features["alsc"] = alsc/len(signal_phasic)
    insc = calculate_insc(signal_phasic)
    features["insc"] = insc
    apsc = calculate_apsc(signal_phasic)
    features["apsc"] = apsc
    rmsc = calculate_rmsc(apsc)
    features["rmsc"] = rmsc
    ilsc = insc/rmsc
    elsc = insc/alsc
    features["ilsc"] = ilsc
    features["elsc"] = elsc
    features["kusc"] = stats.kurtosis(signal_phasic)
    features["sksc"] = stats.skew(signal_phasic)

    features["mosc"] = stats.moment(signal_phasic, 2)

    sig_fft = fft(np.array(signal_phasic))
    freqs, psd = signal.welch(sig_fft, return_onesided=False)
    f1sc = psd[np.where(np.logical_and(freqs > 0.1, freqs < 0.2))]
    f2sc = psd[np.where(np.logical_and(freqs > 0.2, freqs < 0.3))]
    f3sc = psd[np.where(np.logical_and(freqs > 0.3, freqs < 0.4))]

    features["f1sc"] = f1sc.mean()
    features["f2sc"] = f2sc.mean()
    features["f3sc"] = f3sc.mean()

    return features


def from_scl(signal, sr):

    features = {}
    s = np.array(signal)
    features['EDA_mean'] = np.mean(signal)
    features['EDA_std'] = np.std(signal)
    scl_gradient = np.gradient(signal)
    features['EDA_derivative_positive'] = scl_gradient.mean()
    second_derivative=np.gradient(scl_gradient)
    features['EDA_second_derivative'] = second_derivative.mean()
    # scl_features_['EDA_derivative_negative'].append(np.mean(np.take(scl_gradient,np.where(scl_gradient<0))))
    features['EDA_skewness'] = stats.skew(s)
    features['EDA_kurtosis'] = stats.kurtosis(s)

    return features