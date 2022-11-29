import pytest
import numpy as np

from biobss.utils.sample_loader import *
from biobss.ppgtools.ppg_features import *

class TestPpgFeatures(object):

    def test_num_features(self):
        
        #Load the sample data
        data, info = load_sample_data(data_type='PPG')
        sig = np.asarray(data['PPG'])
        fs = info['sampling_rate']

        locs_peaks = np.array([60, 117, 169, 219, 270, 317, 364, 413, 464, 514, 563])
        peaks = sig[locs_peaks]
        locs_onsets = np.array([ 49, 105, 159, 209, 259, 306, 355, 402, 453, 503, 552, 606])
        onsets = sig[locs_onsets]

        features_cycles = from_cycles(sig=sig, peaks_locs=locs_peaks, peaks_amp=peaks, troughs_locs=locs_onsets, troughs_amp=onsets, sampling_rate=fs)
        features_segment = from_segment(sig=sig, sampling_rate=fs)

        assert len(features_cycles) == 24
        assert len(features_segment) == 19


    