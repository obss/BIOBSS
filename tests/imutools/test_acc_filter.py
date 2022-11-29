import pytest
import numpy as np

from biobss.imutools.acc_filter import *
from biobss.utils.sample_loader import *

class TestAccFilter(object):

    def test_signal_lengths(self):

        #Load the sample data
        data, info = load_sample_data(data_type='ACC')
        accx = np.asarray(data['ACCx'])
        accy = np.asarray(data['ACCy'])
        accz = np.asarray(data['ACCz'])
        fs = info['sampling_rate']
        L = info['signal_length']


        filtered_accx = filter_acc(accx, sampling_rate=fs, method='lowpass')
        filtered_accy = filter_acc(accy, sampling_rate=fs, method='lowpass')
        filtered_accz = filter_acc(accz, sampling_rate=fs, method='lowpass')

        assert len(accx) == len(filtered_accx)
        assert len(accy) == len(filtered_accy)
        assert len(accz) == len(filtered_accz)