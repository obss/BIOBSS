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
from .stat_features import *
from .hjorth import *
from .signal_features import *
from .freq_features import *



def from_decomposed(signal_phasic, signal_tonic, sr):

    features={}
    phasic_features = from_scr(signal_phasic)
    tonic_features = from_scl(signal_tonic)
    features.update(phasic_features)
    features.update(tonic_features)
    
    return features


def from_signal(signal, sr=20):

    decomposed_ = eda_decompose(signal, sr)
    eda_phasic = decomposed_['EDA_Phasic']
    eda_tonic = decomposed_['EDA_Tonic']
    features = from_decomposed(eda_phasic, eda_tonic, sr)
    return features

def from_windows(eda_windows,sr=20,parallel=False,n_jobs=6):
    
    if(parallel):
        f_pool=multiprocessing.Pool(processes=n_jobs)
        features=f_pool.map(partial(from_signal,sr=sr),eda_windows)
        features=pd.DataFrame(features)
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


def from_scr(signal):


    scr_features={}
    scr_features.update(get_stat_features(signal,prefix='scr'))
    scr_features.update(get_hjorth_features(signal,prefix='scr'))
    #scr_features.update(get_signal_features(signal,prefix='scr'))
    scr_features.update(get_freq_features(signal,prefix='scr'))

    return scr_features

def from_scl(signal):


    scl_features={}
    scl_features.update(get_stat_features(signal,prefix='scl'))

    return scl_features