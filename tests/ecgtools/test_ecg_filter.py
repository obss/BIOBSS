import numpy as np
import pytest

from biobss.ecgtools.ecg_filter import *
from biobss.utils.sample_loader import *


def test_signal_lengths(load_sample_ecg):

    data, info = load_sample_ecg

    sig = np.asarray(data["ECG"])
    fs = info["sampling_rate"]

    sig_notch = filter_ecg(sig, sampling_rate=fs, method="notch", f_notch=50, quality_factor=0.5)
    sig_pantompkins = filter_ecg(sig, sampling_rate=fs, method="pantompkins")
    sig_hamilton = filter_ecg(sig, sampling_rate=fs, method="hamilton")
    sig_elgendi = filter_ecg(sig, sampling_rate=fs, method="elgendi")

    assert len(sig) == len(sig_notch)
    assert len(sig) == len(sig_pantompkins)
    assert len(sig) == len(sig_hamilton)
    assert len(sig) == len(sig_elgendi)
