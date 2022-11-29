import pytest
import numpy as np

from biobss.preprocess.signal_filter import *
from biobss.utils.sample_loader import *

class TestFilterSignal(object):

    def test_signal_lengths(self):

        #Load the sample data
        data, info = load_sample_data(data_type='PPG')
        sig = np.asarray(data['PPG'])
        fs = info['sampling_rate']

        sig_lowpass = filter_signal(sig=sig, sampling_rate=fs, filter_type='lowpass', N=2, f_upper=5)
        sig_highpass = filter_signal(sig=sig, sampling_rate=fs, filter_type='highpass', N=2, f_lower=0.5)
        sig_bandpass = filter_signal(sig=sig, sampling_rate=fs, filter_type='bandpass', N=2, f_lower=0.5, f_upper=5)

        assert len(sig) == len(sig_lowpass)
        assert len(sig) == len(sig_highpass)
        assert len(sig) == len(sig_bandpass)
