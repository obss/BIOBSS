import pytest
import numpy as np

from biobss.utils.sample_loader import *
from biobss.ecgtools.ecg_features import *


def test_num_features(load_sample_ecg, ecg_Rpeaks, ecg_fiducials):
    
    data, info = load_sample_ecg

    sig = np.asarray(data['ECG'])
    fs = info['sampling_rate']

    features_Rpeaks = from_Rpeaks(sig=sig, peaks_locs=ecg_Rpeaks, sampling_rate=fs)
    features_waves = from_waves(sig=sig, R_peaks=ecg_Rpeaks, fiducials=ecg_fiducials, sampling_rate=fs)

    assert len(features_Rpeaks) == 15
    assert len(features_Rpeaks[0]) == 8
    assert len(features_waves) == 15
    assert len(features_waves[0]) == 31

    