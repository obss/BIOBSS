import numpy as np
import pytest

from biobss.preprocess.signal_detectpeaks import *
from biobss.utils.sample_loader import *


def test_num_peaks(load_sample_ppg):

    data, info = load_sample_ppg

    sig = np.asarray(data["PPG"])
    fs = info["sampling_rate"]

    info_peakdet = peak_detection(sig, sampling_rate=fs, method="peakdet", delta=0.01)
    # info_heartpy = peak_detection(sig, sampling_rate=fs, method='heartpy')
    info_scipy = peak_detection(sig, sampling_rate=fs, method="scipy")

    assert len(info_peakdet["Peak_locs"]) == 13
    assert sum(sig[info_peakdet["Peak_locs"]]) == pytest.approx(13.2572, 0.01)
    # assert len(info_heartpy['Peak_locs']) ==
    # assert sum(info_heartpy['Peaks']) ==
    assert len(info_scipy["Peak_locs"]) == 16
    assert sum(sig[info_scipy["Peak_locs"]]) == pytest.approx(16.22889, 0.01)
