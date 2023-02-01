import numpy as np
import pytest

from biobss.imutools.acc_filter import *
from biobss.utils.sample_loader import *


def test_signal_lengths(load_sample_acc):

    # Load the sample data
    data, info = load_sample_acc

    accx = np.asarray(data["ACCx"])
    accy = np.asarray(data["ACCy"])
    accz = np.asarray(data["ACCz"])
    fs = info["sampling_rate"]

    filtered_accx = filter_acc(accx, sampling_rate=fs, method="lowpass")
    filtered_accy = filter_acc(accy, sampling_rate=fs, method="lowpass")
    filtered_accz = filter_acc(accz, sampling_rate=fs, method="lowpass")

    assert len(accx) == len(filtered_accx)
    assert len(accy) == len(filtered_accy)
    assert len(accz) == len(filtered_accz)
