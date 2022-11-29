import pytest
import numpy as np

from biobss.preprocess.signal_normalize import *
from biobss.utils.sample_loader import *

class TestSignalNormalize(object):

    def test_signal_range(self):

        #Load the sample data
        data, info = load_sample_data(data_type='PPG')
        sig = np.asarray(data['PPG'])
        fs = info['sampling_rate']
        L = info['signal_length']

        sig_zscore = normalize_signal(sig, method='zscore')
        sig_minmax = normalize_signal(sig, 'minmax')

        assert max(sig_minmax) == pytest.approx(1.0, 0.01)
        assert min(sig_minmax) == pytest.approx(0.0, 0.01)
        assert np.mean(sig_zscore) == pytest.approx(0.0, 0.01)
        assert np.std(sig_zscore) == pytest.approx(1.0, 0.01)