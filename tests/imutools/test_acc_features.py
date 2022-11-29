import pytest
import numpy as np

from biobss.imutools.acc_features import *
from biobss.utils.sample_loader import *

class TestAccFeatures(object):

    def test_num_features(self):
        
        #Load the sample data
        data, info = load_sample_data(data_type='ACC')
        accx = np.asarray(data['ACCx'])
        accy = np.asarray(data['ACCy'])
        accz = np.asarray(data['ACCz'])
        fs = info['sampling_rate']

        features = get_acc_features(signals=[accx,accy,accz], signal_names=['ACCx','ACCy','ACCz'], sampling_rate=fs, magnitude=True)

        assert len(features) == 109


    