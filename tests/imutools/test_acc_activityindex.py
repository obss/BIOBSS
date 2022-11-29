import pytest
import numpy as np

from biobss.imutools.acc_activityindex import *
from biobss.utils.sample_loader import *

class TestAccActivityindex(object):

    def test_num_indices(self):
        
        #Load the sample data
        data, info = load_sample_data(data_type='ACC')
        accx = np.asarray(data['ACCx'])
        accy = np.asarray(data['ACCy'])
        accz = np.asarray(data['ACCz'])
        fs = info['sampling_rate']
        L = info['signal_length']
    
        #Calculate activity indices
        pim = calc_activity_index(accx, accy, accz, signal_length=L, sampling_rate=fs, metric='PIM')
        zcm = calc_activity_index(accx, accy, accz, signal_length=L, sampling_rate=fs, metric='ZCM')
        tat = calc_activity_index(accx, accy, accz, signal_length=L, sampling_rate=fs, metric='TAT')
        mad = calc_activity_index(accx, accy, accz, signal_length=L, sampling_rate=fs, metric='MAD')
        enmo = calc_activity_index(accx, accy, accz, signal_length=L, sampling_rate=fs, metric='ENMO')
        hfen = calc_activity_index(accx, accy, accz, signal_length=L, sampling_rate=fs, metric='HFEN')
        ai = calc_activity_index(accx, accy, accz, signal_length=L, sampling_rate=fs, metric='AI', baseline_variance=[0.5,0.5,0.5])
        
        assert len(pim) == 5
        assert len(zcm) == 5
        assert len(tat) == 5
        assert len(mad) == 6
        assert len(enmo) == 1
        assert len(hfen) == 2
        assert len(ai) == 2


    