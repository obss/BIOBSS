import pytest
import numpy as np

from biobss.ppgtools.ppg_peaks import *
from biobss.utils.sample_loader import *
from biobss.preprocess.signal_detectpeaks import peak_detection


def test_num_peaks(load_sample_ppg):

    data, info = load_sample_ppg
    
    sig = np.asarray(data['PPG'])
    fs = info['sampling_rate']
    
    locs_beats = ppg_beats(sig, sampling_rate=fs, method='peakdet', delta=0.005)
    beats=sig[locs_beats]
    info = peak_detection(sig, sampling_rate=fs, method='peakdet', delta=0.01)
    locs_onsets=info['Trough_locs']
    onsets=info['Troughs']

    assert len(locs_beats) == 12
    assert sum(beats) == pytest.approx(12.04455, 0.01)
    assert len(locs_onsets) == 12
    assert sum(onsets) == pytest.approx(11.855229, 0.1)


def test_peak_control(load_sample_ppg, ppg_peaks, ppg_onsets, ppg_irregular):

    data, _ = load_sample_ppg
    
    sig = np.asarray(data['PPG'])

    locs_peaks = ppg_irregular
    locs_onsets = ppg_onsets
    
    expected_locs_peaks = ppg_peaks
    expected_locs_onsets = ppg_onsets
    
    result = peak_control(sig, peaks_locs=locs_peaks, troughs_locs=locs_onsets, type='peak')

    assert len(result['Peak_locs'] ) == len(expected_locs_peaks)
    assert len(result['Trough_locs']) == len(expected_locs_onsets)

    assert all([a == b for a, b in zip(result['Peak_locs'], expected_locs_peaks)])
    assert all([a == b for a, b in zip(result['Trough_locs'], expected_locs_onsets)])

