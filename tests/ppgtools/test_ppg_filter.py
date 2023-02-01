import numpy as np
import pytest

from biobss.ppgtools.ppg_filter import *
from biobss.utils.sample_loader import *


def test_signal_lengths(load_sample_ppg):

    data, info = load_sample_ppg

    sig = np.asarray(data["PPG"])
    fs = info["sampling_rate"]

    sig_bandpass = filter_ppg(sig, sampling_rate=fs, method="bandpass")

    assert len(sig) == len(sig_bandpass)
