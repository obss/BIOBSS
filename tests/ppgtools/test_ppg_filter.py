import pytest
import numpy as np

from biobss.utils.sample_loader import *
from biobss.ppgtools.ppg_filter import *

class TestPpgFilter(object):

    def test_signal_lengths(self):

        #Load the sample data
        data, info = load_sample_data(data_type='PPG')
        sig = np.asarray(data['PPG'])
        fs = info['sampling_rate']
        L = info['signal_length']

        sig_bandpass = filter_ppg(sig, sampling_rate=fs, method='bandpass')

        assert len(sig) == len(sig_bandpass)


