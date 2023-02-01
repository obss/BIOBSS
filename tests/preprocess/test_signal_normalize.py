import numpy as np
import pytest

from biobss.preprocess.signal_normalize import *
from biobss.utils.sample_loader import *


def test_signal_range(load_sample_ppg):

    data, _ = load_sample_ppg

    sig = np.asarray(data["PPG"])

    sig_zscore = normalize_signal(sig, method="zscore")
    sig_minmax = normalize_signal(sig, "minmax")

    assert max(sig_minmax) == pytest.approx(1.0, 0.01)
    assert min(sig_minmax) == pytest.approx(0.0, 0.01)
    assert np.mean(sig_zscore) == pytest.approx(0.0, 0.01)
    assert np.std(sig_zscore) == pytest.approx(1.0, 0.01)
