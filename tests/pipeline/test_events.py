import numpy as np
import pandas as pd
import pytest

from biobss.ecgtools.ecg_peaks import ecg_detectpeaks
from biobss.pipeline.bio_data import Bio_Data
from biobss.pipeline.event_channel import Event_Channel
from biobss.pipeline.event_input import *


@pytest.fixture()
def sample_peaks(sample_ecg_array):
    peaks = ecg_detectpeaks(sample_ecg_array, sampling_rate=256)
    return peaks


@pytest.fixture()
def sample_channel(sample_peaks):
    return Event_Channel(sample_peaks, name="ecg_peaks", sampling_rate=256)


@pytest.fixture()
def ref_peak_bio_data(sample_channel):
    ref_data = Bio_Data()
    ref_data.add_channel(sample_channel)
    return ref_data


def test_a_channel_input(sample_channel, ref_peak_bio_data):
    b_data = convert_event(sample_channel, "ecg_peaks")
    assert b_data == ref_peak_bio_data


def test_b_dict_input(sample_peaks, ref_peak_bio_data):
    sample_dict = {"ecg_peaks": sample_peaks}
    b_data = convert_event(sample_dict, sampling_rate=256)
    assert b_data == ref_peak_bio_data


def test_c_array_input(sample_peaks, ref_peak_bio_data):
    b_data = convert_event(sample_peaks, sampling_rate=256, name="ecg_peaks")
    assert b_data == ref_peak_bio_data


def test_d_list_input(sample_peaks, ref_peak_bio_data):
    sample_list = list(sample_peaks)
    b_data = convert_event(sample_list, sampling_rate=256, name="ecg_peaks")
    assert b_data == ref_peak_bio_data


def test_e_data_frame_input(sample_peaks, ref_peak_bio_data):
    sample_data_frame = pd.DataFrame(np.transpose([sample_peaks]), columns=["ecg_peaks"])
    b_data = convert_event(sample_data_frame, sampling_rate=256, name="ecg_peaks")
    assert b_data == ref_peak_bio_data
