import pytest
import numpy as np

from biobss.resptools.resp_estimation import *
from biobss.utils.sample_loader import *

class TestRespEstimation(object):

    def test_signal_lengths(self):

        #Load the sample data
        data, info = load_sample_data(data_type='PPG')
        sig = np.asarray(data['PPG'])
        fs = info['sampling_rate']

        locs_peaks = np.array([60, 117, 169, 219, 270, 317, 364, 413, 464, 514, 563])
        peaks = np.array([1.0201, 1.016, 1.0211, 1.0224, 1.0208, 1.019, 1.0201, 1.0213, 1.0215, 1.019, 1.02])
        locs_onsets = np.array([ 49, 105, 159, 209, 259, 306, 355, 402, 453, 503, 552, 606])
        onsets = np.array([0.98663, 0.98157, 0.98788, 0.98706, 0.98508, 0.98593, 0.98648, 0.98541, 0.98416, 0.98499, 0.98505, 0.98499])

        info=extract_resp_sig(peaks_locs=locs_peaks,peaks_amp=peaks,troughs_amp=onsets,sampling_rate=fs,mod_type=['AM','FM','BW'],resampling_rate=10)
       
        x_am=info['am_x']
        x_fm=info['fm_x']
        x_bw=info['bw_x']

        assert len(x_am) == 84
        assert len(x_fm) == 76
        assert len(x_bw) == 84

