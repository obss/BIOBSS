import numpy as np
import pandas as pd
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
from numpy.typing import ArrayLike

def from_decomposed(signal_phasic:ArrayLike, signal_tonic:ArrayLike, sr:float)->dict:
    """Method to calculate features over Tonic and Phasic EDA components

    Args:
        signal_phasic (1-D): Phasic component of EDA
        signal_tonic (arraylike): Tonic component of EDA
        sr (float): Sampling Rate

    Returns:
        dictionary: dictionary of EDA features
    """

    features = {}
    phasic_features = from_scr(signal_phasic)
    tonic_features = from_scl(signal_tonic)
    features.update(phasic_features)
    features.update(tonic_features)

    return features


def from_signal(signal:ArrayLike, sr=20.)->dict:
    """Method to calculate features over EDA signal

    Args:
        signal (arraylike): EDA Signal
        sr (float): Sampling Rate. Defaults to 20.

    Returns:
        dictionary: dictionary of EDA features
    """

    decomposed_ = eda_decompose(signal, sr)
    eda_phasic = decomposed_["EDA_Phasic"]
    eda_tonic = decomposed_["EDA_Tonic"]
    features = from_decomposed(eda_phasic, eda_tonic, sr)
    return features


def from_windows(eda_windows:ArrayLike, sr=20., parallel=False, n_jobs=6)->pd.DataFrame:
    """Method to calculate EDA features over set of EDA signals

    Args:
        eda_windows (2-D array): Set of EDA signals (Windows)
        sr (float, optional): Sampling Rate. Defaults to 20.
        parallel (bool, optional): Whether to process parallely. Defaults to False.
        n_jobs (int, optional): Number of jobs used in parallel processing. Defaults to 6.

    Returns:
        DataFrame: EDA features of given windows
    """

    if parallel:
        f_pool = multiprocessing.Pool(processes=n_jobs)
        features = f_pool.map(partial(from_signal, sr=sr), eda_windows)
        features = pd.DataFrame(features)
    else:
        features = []
        for w in eda_windows:
            features.append(from_signal(w, sr))
        features = pd.DataFrame(features)
    return features


def from_decomposed_windows(
    phasic_windows:ArrayLike, tonic_windows:ArrayLike, sr, parallel=False, n_jobs=6
)->pd.DataFrame:
    """Method to calculate EDA features over set of decomposed EDA signals

    Args:
        phasic_windows (2-D array): set of phasic eda signals
        tonic_windows (2-D array): set of tonic eda signals
        sr (float): sampling rate
        parallel (bool, optional): Whether to process parallely. Defaults to False.
        n_jobs (int, optional): Number of jobs used in parallel processing. Defaults to 6.

    Returns:
        DataFrame: EDA features of given windows
    """

    scr_features = []
    scl_features = []
    if parallel:
        f_pool = multiprocessing.Pool(processes=n_jobs)
        scr_features = f_pool.map(partial(from_scr, sr=sr), phasic_windows)
        f_pool = multiprocessing.Pool(processes=n_jobs)
        scl_features = f_pool.map(partial(from_scl, sr=sr), tonic_windows)

    else:
        for scrw in phasic_windows:
            scr_features.append(from_scr(scrw, sr))

        for sclw in tonic_windows:
            scl_features.append(from_scl(sclw, sr))

    scr_features = pd.DataFrame(scr_features)
    scl_features = pd.DataFrame(scl_features)
    return pd.concat([scr_features, scl_features], axis=0, ignore_index=True)


def from_scr(signal:ArrayLike)->dict:
    """Calculate features over Phasic EDA signal

    Args:
        signal (arraylike): Phasic EDA signal

    Returns:
        dict: SCR features over Phasic EDA
    """

    scr_features = {}
    scr_features.update(get_stat_features(signal, prefix="scr"))
    scr_features.update(get_hjorth_features(signal, prefix="scr"))
    scr_features.update(get_signal_features(signal, prefix="scr"))
    scr_features.update(get_freq_features(signal, prefix="scr"))

    return scr_features


def from_scl(signal:ArrayLike)->dict:
    """Calculate features over Tonic EDA signal

    Args:
        signal (arraylike): Tonic EDA signal

    Returns:
        dict: SCL features over Tonic EDA
    """

    scl_features = {}
    scl_features.update(get_stat_features(signal, prefix="scl"))

    return scl_features
