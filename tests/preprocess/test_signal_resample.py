import pytest
import numpy as np

from biobss.preprocess.signal_resample import *
from biobss.utils.sample_loader import *

class TestSignalResample(object):

    def test_signal_length(self):

        #Load the sample data
        data, info = load_sample_data(data_type='PPG')
        sig = np.asarray(data['PPG'])
        fs = info['sampling_rate']
        L = info['signal_length']

        f_rs = fs / 10

        resampled = resample_signal(sig, sampling_rate=fs, target_sampling_rate=f_rs)

        assert len(resampled) == f_rs * L
