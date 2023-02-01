from __future__ import annotations

import multiprocessing
from functools import partial

import pandas as pd
from numpy.typing import ArrayLike

from .eda_decompose import *
from .eda_freqdomain import *
from .eda_hjorth import *
from .eda_signalfeatures import *
from .eda_statistical import *


def from_decomposed(signal_phasic: ArrayLike, signal_tonic: ArrayLike, sampling_rate: float) -> dict:
    """Calculates features over Tonic and Phasic EDA components

    Args:
        signal_phasic (ArrayLike): Phasic component of EDA signal.
        signal_tonic (ArrayLike): Tonic component of EDA signal.
        sampling_rate (float): Sampling rate of the EDA signal (Hz).

    Returns:
        dict: Dictionary of calculated features.
    """
    features = {}
    phasic_features = from_scr(signal_phasic)
    tonic_features = from_scl(signal_tonic)
    features.update(phasic_features)
    features.update(tonic_features)

    return features


def from_signal(signal: ArrayLike, sampling_rate: float = 20.0) -> dict:
    """Calculates features over EDA signal.

    Args:
        signal (ArrayLike): EDA signal.
        sampling_rate (float, optional): Sampling rate of the EDA signal (Hz). Defaults to 20.0 Hz.

    Returns:
        dict: Dictionary of calculated features.
    """
    decomposed_ = eda_decompose(signal, sampling_rate)
    eda_phasic = decomposed_["EDA_Phasic"]
    eda_tonic = decomposed_["EDA_Tonic"]
    features = from_decomposed(eda_phasic, eda_tonic, sampling_rate)

    return features


def from_windows(
    eda_windows: ArrayLike, sampling_rate: float = 20.0, parallel: bool = False, n_jobs: int = 6
) -> pd.DataFrame:
    """Calculates EDA features over set of EDA signals.

    Args:
        eda_windows (ArrayLike): Set of EDA signals (Windows).
        sampling_rate (float, optional): Sampling rate of the EDA signals (Hz). Defaults to 20.0 Hz.
        parallel (bool, optional): Whether to process parallely. Defaults to False.
        n_jobs (int, optional): Number of jobs used in parallel processing. Defaults to 6.

    Returns:
        pd.DataFrame: EDA features of given windows.
    """
    if parallel:
        f_pool = multiprocessing.Pool(processes=n_jobs)
        features = f_pool.map(partial(from_signal, sr=sampling_rate), eda_windows)
        features = pd.DataFrame(features)
    else:
        features = []
        for w in eda_windows:
            features.append(from_signal(w, sampling_rate))
        features = pd.DataFrame(features)

    return features


def from_decomposed_windows(
    phasic_windows: ArrayLike, tonic_windows: ArrayLike, sampling_rate: float, parallel: bool = False, n_jobs: int = 6
) -> pd.DataFrame:
    """Calculates EDA features over set of decomposed EDA signals.

    Args:
        phasic_windows (ArrayLike): Set of phasic eda signals
        tonic_windows (ArrayLike): Set of tonic eda signals
        sampling_rate (float): Sampling rate of the EDA signal (Hz).
        parallel (bool, optional): Whether to process parallely. Defaults to False.
        n_jobs (int, optional): Number of jobs used in parallel processing. Defaults to 6.

    Returns:
        pd.DataFrame: EDA features of given windows
    """
    scr_features = []
    scl_features = []
    if parallel:
        f_pool = multiprocessing.Pool(processes=n_jobs)
        scr_features = f_pool.map(partial(from_scr, sr=sampling_rate), phasic_windows)
        f_pool = multiprocessing.Pool(processes=n_jobs)
        scl_features = f_pool.map(partial(from_scl, sr=sampling_rate), tonic_windows)

    else:
        for scrw in phasic_windows:
            scr_features.append(from_scr(scrw, sampling_rate))

        for sclw in tonic_windows:
            scl_features.append(from_scl(sclw, sampling_rate))

    scr_features = pd.DataFrame(scr_features)
    scl_features = pd.DataFrame(scl_features)

    return pd.concat([scr_features, scl_features], axis=0, ignore_index=True)


def from_scr(signal: ArrayLike) -> dict:
    """Calculates features over Phasic EDA signal.

    Args:
        signal (ArrayLike): Phasic component of EDA signal.

    Returns:
        dict: SCR features over Phasic EDA signal.
    """
    scr_features = {}
    scr_features.update(eda_stat_features(signal, prefix="scr"))
    scr_features.update(eda_hjorth_features(signal, prefix="scr"))
    scr_features.update(eda_signal_features(signal, prefix="scr"))
    scr_features.update(eda_freq_features(signal, prefix="scr"))

    return scr_features


def from_scl(signal: ArrayLike) -> dict:
    """Calculates features over Tonic EDA signal.

    Args:
        signal (ArrayLike): Tonic component of EDA signal.

    Returns:
        dict: SCL features over Tonic EDA signal.
    """
    scl_features = {}
    scl_features.update(eda_stat_features(signal, prefix="scl"))

    return scl_features
