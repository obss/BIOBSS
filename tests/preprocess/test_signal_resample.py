import numpy as np
import pytest

from biobss.preprocess.signal_resample import *
from biobss.utils.sample_loader import *


def test_signal_length(load_sample_ppg):

    data, info = load_sample_ppg

    sig = np.asarray(data["PPG"])
    fs = info["sampling_rate"]
    L = info["signal_length"]

    f_rs = fs / 10

    resampled = resample_signal(sig, sampling_rate=fs, target_sampling_rate=f_rs)

    assert len(resampled) == f_rs * L
