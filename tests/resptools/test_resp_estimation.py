import numpy as np
import pytest

from biobss.resptools.resp_estimation import *
from biobss.utils.sample_loader import *


def test_signal_lengths(load_sample_ppg, ppg_peaks, ppg_onsets):

    data, info = load_sample_ppg
    sig = np.asarray(data["PPG"])
    fs = info["sampling_rate"]

    info = extract_resp_sig(
        sig=sig,
        peaks_locs=ppg_peaks,
        troughs_locs=ppg_onsets,
        sampling_rate=fs,
        mod_type=["AM", "FM", "BW"],
        resampling_rate=10,
    )

    x_am = info["am_x"]
    x_fm = info["fm_x"]
    x_bw = info["bw_x"]

    assert len(x_am) == 84
    assert len(x_fm) == 76
    assert len(x_bw) == 84
