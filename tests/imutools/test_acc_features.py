import numpy as np
import pytest

from biobss.imutools.acc_features import *
from biobss.utils.sample_loader import *


def test_num_features(load_sample_acc):

    # Load the sample data
    data, info = load_sample_acc

    accx = np.asarray(data["ACCx"])
    accy = np.asarray(data["ACCy"])
    accz = np.asarray(data["ACCz"])
    fs = info["sampling_rate"]

    features = get_acc_features(
        signals=[accx, accy, accz], signal_names=["ACCx", "ACCy", "ACCz"], sampling_rate=fs, magnitude=True
    )

    assert len(features) == 147
