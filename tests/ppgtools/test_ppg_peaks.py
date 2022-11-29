import pytest
import numpy as np

from biobss.ppgtools.ppg_peaks import *
from biobss.utils.sample_loader import *
from biobss.preprocess.signal_detectpeaks import peak_detection

class TestPpgPeaks(object):

    def test_num_peaks(self):

        #Load the sample data
        data, info = load_sample_data(data_type='PPG')
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

    
    def test_peak_control(self):

        #Load the sample data
        data, info = load_sample_data(data_type='PPG')
        sig = np.asarray(data['PPG'])
        fs = info['sampling_rate']

        locs_peaks = np.array([0, 5, 60, 117, 130, 169, 219, 270, 290, 317, 364, 413, 464, 514, 563, 618, 630])
        locs_onsets = np.array([49, 105, 159, 209, 259, 306, 355, 402, 453, 503, 552, 606])
        

        expected_locs_peaks = np.array([60, 117, 169, 219, 270, 317, 364, 413, 464, 514, 563])
        expected_locs_onsets = np.array([49, 105, 159, 209, 259, 306, 355, 402, 453, 503, 552, 606])

        result = peak_control(sig, peaks_locs=locs_peaks, troughs_locs=locs_onsets, type='peak')

        assert len(result['Peak_locs'] ) == len(expected_locs_peaks)
        assert len(result['Trough_locs']) == len(expected_locs_onsets)

        assert all([a == b for a, b in zip(result['Peak_locs'], expected_locs_peaks)])
        assert all([a == b for a, b in zip(result['Trough_locs'], expected_locs_onsets)])

    