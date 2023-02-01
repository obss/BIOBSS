import numpy as np
import pytest

from biobss.ppgtools.apg_features import *
from biobss.preprocess.signal_filter import *
from biobss.utils.sample_loader import *


def test_num_features(load_sample_ppg, ppg_onsets, apg_fiducials):

    data, info = load_sample_ppg

    sig = np.asarray(data["PPG"])
    fs = info["sampling_rate"]

    filtered_sig = filter_signal(sig=sig, sampling_rate=fs, signal_type="PPG", method="bandpass")

    vpg_sig = np.gradient(filtered_sig) / (1 / fs)
    apg_sig = np.gradient(vpg_sig) / (1 / fs)

    apg_features = get_apg_features(apg_sig=apg_sig, locs_O=ppg_onsets, fiducials=apg_fiducials, sampling_rate=fs)

    assert len(apg_features) == 18
