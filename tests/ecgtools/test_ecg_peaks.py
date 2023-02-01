import numpy as np
import pytest

from biobss.ecgtools.ecg_peaks import *
from biobss.utils.sample_loader import *


def test_num_Rpeaks(load_sample_ecg):

    data, info = load_sample_ecg

    sig = np.asarray(data["ECG"])
    fs = info["sampling_rate"]

    locs_pantompkins = ecg_detectpeaks(sig, sampling_rate=fs, method="pantompkins")
    peaks_pantompkins = sig[locs_pantompkins]

    locs_hamilton = ecg_detectpeaks(sig, sampling_rate=fs, method="hamilton")
    peaks_hamilton = sig[locs_hamilton]

    locs_elgendi = ecg_detectpeaks(sig, sampling_rate=fs, method="elgendi")
    peaks_elgendi = sig[locs_elgendi]

    assert len(locs_pantompkins) == 15
    assert sum(peaks_pantompkins) == pytest.approx(-1.26, 0.01)
    assert len(locs_hamilton) == 17
    assert sum(peaks_hamilton) == pytest.approx(-0.76, 0.01)
    assert len(locs_elgendi) == 15
    assert sum(peaks_elgendi) == pytest.approx(-0.53, 0.01)
