import numpy as np
import pytest

from biobss.pipeline.bio_channel import Channel
from biobss.pipeline.bio_data import Bio_Data


def test_a_creation(sample_ecg_array):
    channel = Channel(sample_ecg_array, name="ecg", sampling_rate=256)
    assert True


def test_b_creation_windowed(sample_windowed):
    channel = Channel(sample_windowed, name="ecg", sampling_rate=256)
    assert True


def test_c_content(sample_ecg_array):
    channel = Channel(sample_ecg_array, name="ecg", sampling_rate=256)
    assert np.all(channel.channel == sample_ecg_array)


def test_d_content_windowed(sample_windowed):
    channel = Channel(sample_windowed, name="ecg", sampling_rate=256)
    assert np.all(channel.channel == sample_windowed)
