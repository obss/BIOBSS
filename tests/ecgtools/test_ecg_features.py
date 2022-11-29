import pytest
import numpy as np

from biobss.utils.sample_loader import *
from biobss.ecgtools.ecg_features import *

class TestEcgFeatures(object):

    def test_num_features(self):
        
        #Load the sample data
        data, info = load_sample_data(data_type='ECG')
        sig = np.asarray(data['ECG'])
        fs = info['sampling_rate']

        locs_peaks = [94, 259, 430, 595, 758, 935, 1101, 1263, 1427, 1593, 1760, 1914, 2087, 2253, 2422]
        fiducials = {'ECG_P_Peaks': [61, 225, 390, 555, 726, 894, 1061, 1223, 1388, 1552, 1720, 1882, 2047, 2214, 2382],
                     'ECG_Q_Peaks': [79, 247, 408, 573, 750, 915, 1077, 1238, 1406, 1571, 1735, 1900, 2065, 2226, 2401],
                     'ECG_S_Peaks': [102,267,438,660,766,956,1107,1305,1453,1615,1826,1922,2156,2324,2492],
                     'ECG_T_Peaks': [148, 316, 474, 689, 813, 981, 1146, 1361, 1475, 1639, 1855, 1966, 2184, 2332, 2524],
                     'ECG_P_Onsets': [51, 215, 378, 544, 715, 883, 1049, 1211, 1377, 1542, 1708, 1871, 2036, 2202, 2371],
                     'ECG_T_Offsets': [164, 329, 490, 696, 827, 998, 1165, 1367, 1491, 1654, 1861, 1983, 2191, 2337, 2525]
        }
        features_Rpeaks = from_Rpeaks(sig=sig, peaks_locs=locs_peaks, sampling_rate=fs)
        features_waves = from_waves(sig=sig, R_peaks=locs_peaks, fiducials=fiducials, sampling_rate=fs)

        assert len(features_Rpeaks) == 15
        assert len(features_Rpeaks[0]) == 8
        assert len(features_waves) == 15
        assert len(features_waves[0]) == 31

    