import pytest
import numpy as np

from biobss.hrvtools.hrv_features import *
from biobss.utils.sample_loader import *

class TestHrvFeatures(object):

    def test_num_features(self):

        #Load the sample data
        data, info = load_sample_data(data_type='ECG')
        sig = np.asarray(data['ECG'])
        fs = info['sampling_rate']        
        L = info['signal_length']

        locs_peaks = [94, 259, 430, 595, 758, 935, 1101, 1263, 1427, 1593, 1760, 1914, 2087, 2253, 2422]

        features = get_hrv_features(sampling_rate=fs, signal_length=L, signal_type='ECG', input_type='peaks', peaks_locs=locs_peaks)

        assert len(features) == 39  