import numpy as np
import pytest

from biobss.hrvtools.hrv_features import *
from biobss.utils.sample_loader import *


def test_num_features(load_sample_ecg, ecg_Rpeaks):

    _, info = load_sample_ecg

    fs = info["sampling_rate"]
    L = info["signal_length"]

    features = get_hrv_features(
        sampling_rate=fs, signal_length=L, signal_type="ECG", input_type="peaks", peaks_locs=ecg_Rpeaks
    )

    assert len(features) == 39
