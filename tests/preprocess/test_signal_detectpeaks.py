import pytest
import numpy as np

from biobss.preprocess.signal_detectpeaks import *
from biobss.utils.sample_loader import *

class TestPeakDetection(object):

    def test_num_peaks(self):

        #Load the sample data
        data, info = load_sample_data(data_type='PPG')
        sig = np.asarray(data['PPG'])
        fs = info['sampling_rate']

        info_peakdet = peak_detection(sig, sampling_rate=fs, method='peakdet', delta=0.01)
        #info_heartpy = peak_detection(sig, sampling_rate=fs, method='heartpy')
        info_scipy = peak_detection(sig, sampling_rate=fs, method='scipy')


        assert len(info_peakdet['Peak_locs']) == 13
        assert sum(info_peakdet['Peaks']) == pytest.approx(13.2572, 0.01)
        #assert len(info_heartpy['Peak_locs']) == 
        #assert sum(info_heartpy['Peaks']) == 
        assert len(info_scipy['Peak_locs']) == 16
        assert sum(info_scipy['Peaks']) == pytest.approx(16.22889, 0.01)

      