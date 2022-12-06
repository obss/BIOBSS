import pytest
import numpy as np

from biobss.utils.sample_loader import *
from biobss.ppgtools.ppg_features import *


def test_num_features(load_sample_ppg, ppg_peaks, ppg_onsets):
    
    data, info = load_sample_ppg
    
    sig = np.asarray(data['PPG'])
    fs = info['sampling_rate']

    peaks = sig[ppg_peaks]
    onsets = sig[ppg_onsets]

    features_cycles = from_cycles(sig=sig, peaks_locs=ppg_peaks, peaks_amp=peaks, troughs_locs=ppg_onsets, troughs_amp=onsets, sampling_rate=fs)
    features_segment = from_segment(sig=sig, sampling_rate=fs)

    assert len(features_cycles) == 24
    assert len(features_segment) == 19


    