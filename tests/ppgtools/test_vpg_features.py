import numpy as np
import pytest

from biobss.ppgtools.vpg_features import *
from biobss.utils.sample_loader import *


def test_num_features(load_sample_ppg, ppg_onsets, vpg_fiducials):

    data, info = load_sample_ppg

    sig = np.asarray(data["PPG"])
    fs = info["sampling_rate"]

    vpg_sig = np.gradient(sig) / (1 / fs)

    vpg_features = get_vpg_features(vpg_sig=vpg_sig, locs_O=ppg_onsets, fiducials=vpg_fiducials, sampling_rate=fs)

    assert len(vpg_features) == 7
