import numpy as np
import pytest

from biobss.ppgtools.ppg_features import *
from biobss.preprocess.signal_filter import *
from biobss.utils.sample_loader import *


def test_num_features(load_sample_ppg, ppg_peaks, ppg_onsets, ppg_fiducials):

    data, info = load_sample_ppg

    sig = np.asarray(data["PPG"])
    fs = info["sampling_rate"]

    features_cycles = from_cycles(
        sig=sig,
        peaks_locs=ppg_peaks,
        troughs_locs=ppg_onsets,
        sampling_rate=fs,
        feature_types=["Time", "Stat"],
        fiducials=ppg_fiducials,
        prefix="ppg",
    )
    features_segment = from_segment(sig=sig, sampling_rate=fs)

    assert len(features_cycles) == 37
    assert len(features_segment) == 19
