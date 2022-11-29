import pytest
import numpy as np

from biobss.ecgtools.ecg_peaks import *
from biobss.utils.sample_loader import *

class TestEcgPeaks(object):

    def test_num_Rpeaks(self):

        #Load the sample data
        data, info = load_sample_data(data_type='ECG')
        sig = np.asarray(data['ECG'])
        fs = info['sampling_rate']

        locs_pantompkins = ecg_peaks(sig, sampling_rate=fs, method='pantompkins')
        peaks_pantompkins = sig[locs_pantompkins]

        locs_hamilton = ecg_peaks(sig, sampling_rate=fs, method='hamilton')
        peaks_hamilton = sig[locs_hamilton]

        locs_elgendi = ecg_peaks(sig, sampling_rate=fs, method='elgendi')
        peaks_elgendi = sig[locs_elgendi]

        assert len(locs_pantompkins) == 15
        assert sum(peaks_pantompkins) == pytest.approx(-1.26, 0.01)
        assert len(locs_hamilton) == 17
        assert sum(peaks_hamilton) == pytest.approx(-0.76, 0.01)
        assert len(locs_elgendi) == 15
        assert sum(peaks_elgendi) == pytest.approx(-0.53, 0.01)

    def test_num_fiducials(self):

        #Load the sample data
        data, info = load_sample_data(data_type='ECG')
        sig = np.asarray(data['ECG'])
        fs = info['sampling_rate']

        fiducials = ecg_waves(sig=sig, sampling_rate=fs, delineator='neurokit2')

        assert len(fiducials) == 6
        assert len(fiducials['ECG_P_Peaks']) == 15
        assert len(fiducials['ECG_Q_Peaks']) == 15
        assert len(fiducials['ECG_S_Peaks']) == 15
        assert len(fiducials['ECG_T_Peaks']) == 15
        assert len(fiducials['ECG_P_Onsets']) == 15
        assert len(fiducials['ECG_T_Offsets'] ) == 15            