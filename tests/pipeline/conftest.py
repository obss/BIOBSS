import numpy as np
import pytest

from biobss.pipeline.bio_channel import Channel
from biobss.pipeline.bio_data import Bio_Data
from biobss.preprocess import segment_signal


@pytest.fixture(scope="module")
def sample_ecg_array(load_sample_ecg):
    sample_data_ = np.array(load_sample_ecg[0])
    sample_data_ = sample_data_.flatten()
    return sample_data_


@pytest.fixture(scope="module")
def ref_ecg_channel(sample_ecg_array):

    gold_channel_ = Channel(sample_ecg_array, name="ecg", sampling_rate=256)
    return gold_channel_


@pytest.fixture(scope="module")
def ref_bio_data(ref_ecg_channel):
    test_data = Bio_Data()
    test_data.add_channel(ref_ecg_channel)
    return test_data


@pytest.fixture(scope="module")
def sample_ecg_dict(sample_ecg_array):
    sample_data_ = {"ecg": sample_ecg_array}
    return sample_data_


@pytest.fixture(scope="module")
def sample_windowed(sample_ecg_array):
    test_data = segment_signal(sample_ecg_array, sampling_rate=256, window_size=5, step_size=1)
    return test_data


@pytest.fixture(scope="module")
def sample_peaks():
    pass


@pytest.fixture(scope="module")
def sample_eda_channel(load_sample_eda):
    eda_data = load_sample_eda[0]
    eda_data = np.array(eda_data)
    eda_data = eda_data.flatten()
    eda_channel = Channel(eda_data, "eda_raw", sampling_rate=700)
    return eda_channel
