import numpy as np
import pytest

from biobss.pipeline.bio_channel import Channel
from biobss.pipeline.bio_data import Bio_Data
from biobss.pipeline.channel_input import convert_channel
from biobss.pipeline.event_input import convert_event


def test_a_join(ref_bio_data):
    ref_bio_data = ref_bio_data.copy()
    try:
        ref_bio_data = ref_bio_data.join(ref_bio_data)
    except Exception as e:
        pytest.fail("Test 1 (Bio_Data Join) failed: {}".format(e))


def test_b_from_dict(ref_bio_data, sample_ecg_dict):
    result = convert_channel(sample_ecg_dict, 256)
    try:
        assert result == ref_bio_data
    except AssertionError as e:
        pytest.fail("Test 2 (from_dict) failed: {}".format(e))


def test_c_from_array(ref_bio_data, sample_ecg_array):

    result = convert_channel(sample_ecg_array, sampling_rate=256, name="ecg")
    try:
        assert result == ref_bio_data
    except AssertionError as e:
        pytest.fail("Test 3 (from_array) failed: {}".format(e))


def test_d_from_channel(ref_bio_data, ref_ecg_channel):
    result = convert_channel(ref_ecg_channel)
    try:
        assert result == ref_bio_data
    except AssertionError as e:
        pytest.fail("Test 4 (from_channel) failed: {}".format(e))


def test_e_from_bio_data(ref_bio_data):
    result = convert_channel(ref_bio_data)
    try:
        assert result == ref_bio_data
    except AssertionError as e:
        pytest.fail("Test 5 (from_bio_data) failed: {}".format(e))


def test_f_from_tuple(ref_bio_data, sample_ecg_array):
    result = convert_channel((sample_ecg_array, "garbage"), sampling_rate=256, name="ecg", index=0)
    try:
        assert result == ref_bio_data
    except AssertionError as e:
        pytest.fail("Test 6 (from_tuple) failed: {}".format(e))


def test_f_from_tuple_b(ref_bio_data, sample_ecg_array):
    result = convert_channel(("garbage", sample_ecg_array, "garbage"), sampling_rate=256, name="ecg", index=1)
    try:
        assert result == ref_bio_data
    except AssertionError as e:
        pytest.fail("Test 6 (from_tuple) failed: {}".format(e))


def test_g_from_list(ref_bio_data, sample_ecg_array):
    result = convert_channel([sample_ecg_array, "garbage"], sampling_rate=256, name="ecg", index=0)
    try:
        assert result == ref_bio_data
    except AssertionError as e:
        pytest.fail("Test 7 (from_list) failed: {}".format(e))


def test_g_from_multiple_dict(sample_ecg_array):
    sample_array = {"ecg": sample_ecg_array, "ecg2": sample_ecg_array}
    result = convert_channel(sample_array, [256, 256])
    try:
        assert result.get_channel_names() == ["ecg", "ecg2"]
    except AssertionError as e:
        pytest.fail("Test 8 (from_multiple_dict) failed: {}".format(e))


def test_i_from_multiple_channel(ref_ecg_channel):
    sample_array = [ref_ecg_channel.copy(), ref_ecg_channel.copy()]
    result = convert_channel(sample_array, [256, 256], name=["ecg", "ecg2"])
    try:
        assert result.get_channel_names() == ["ecg", "ecg2"]
    except AssertionError as e:
        pytest.fail("Test 10 (from_multiple_channel) failed: {}".format(e))


def test_j_from_windowed(sample_windowed):
    result = convert_channel(sample_windowed, sampling_rate=256, name="ecg", n_windows=sample_windowed.shape[0])
    other = Channel(sample_windowed, name="ecg", sampling_rate=256)
    other_bd = Bio_Data()
    other_bd.add_channel(other)
    try:
        assert result == other_bd
    except AssertionError as e:
        pytest.fail("Test 11 (from_windowed) failed: {}".format(e))
