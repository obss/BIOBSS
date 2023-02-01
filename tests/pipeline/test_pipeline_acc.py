import sys

import numpy as np
import pandas as pd
import pytest

from biobss.imutools.acc_correlation import *
from biobss.imutools.acc_features import *
from biobss.imutools.acc_freqdomain import *
from biobss.imutools.acc_statistical import *
from biobss.pipeline.bio_channel import Channel
from biobss.pipeline.bio_data import Bio_Data
from biobss.pipeline.bio_process import Bio_Process
from biobss.pipeline.channel_input import *
from biobss.pipeline.pipeline import Bio_Pipeline
from biobss.preprocess.signal_normalize import normalize_signal
from biobss.preprocess.signal_resample import resample_signal


@pytest.fixture()
def sample_acc(load_sample_acc):
    sample = load_sample_acc[0]
    return sample


@pytest.fixture()
def gold_channel_x(sample_acc):
    accx = np.asarray(
        sample_acc["ACCx"],
    )
    gold_channel_ = Channel(accx, name="accx", sampling_rate=32)
    return gold_channel_


@pytest.fixture()
def gold_channel_y(sample_acc):
    accy = np.asarray(sample_acc["ACCy"])
    gold_channel_ = Channel(accy, name="accy", sampling_rate=32)
    return gold_channel_


@pytest.fixture()
def gold_channel_z(sample_acc):
    accz = np.asarray(sample_acc["ACCz"])
    gold_channel_ = Channel(accz, name="accz", sampling_rate=32)
    return gold_channel_


@pytest.fixture()
def gold_bio_data(sample_acc):
    test_data = sample_acc
    accx = np.asarray(test_data["ACCx"])
    accy = np.asarray(test_data["ACCy"])
    accz = np.asarray(test_data["ACCz"])
    gold_channel_x = Channel(accx, name="accx", sampling_rate=32)
    gold_channel_y = Channel(accy, name="accy", sampling_rate=32)
    gold_channel_z = Channel(accz, name="accz", sampling_rate=32)
    gold_bio_data_ = Bio_Data()
    gold_bio_data_.add_channel(gold_channel_x)
    gold_bio_data_.add_channel(gold_channel_y)
    gold_bio_data_.add_channel(gold_channel_z)
    return gold_bio_data_


@pytest.fixture()
def sample_data(sample_acc):
    test_data = sample_acc
    accx = np.asarray(test_data["ACCx"])
    accy = np.asarray(test_data["ACCy"])
    accz = np.asarray(test_data["ACCz"])
    sample_data_ = {"accx": accx, "accy": accy, "accz": accz}
    return sample_data_


@pytest.fixture()
def sample_array_x(sample_acc):
    test_data = sample_acc
    accx = np.array(test_data["ACCx"])
    return accx


@pytest.fixture()
def sample_array_y(sample_acc):
    test_data = sample_acc
    accy = np.array(test_data["ACCy"])
    return accy


@pytest.fixture()
def sample_array_z(sample_acc):
    test_data = sample_acc
    accz = np.array(test_data["ACCz"])
    return accz


def test_a_creation():
    pipeline = Bio_Pipeline()
    assert True


def test_b_creation_windowed():
    pipeline = Bio_Pipeline(windowed_process=True, window_size=10, step_size=10)
    assert True


def test_c_set_input_from_channel(gold_channel_x, gold_channel_y, gold_channel_z):
    pipeline = Bio_Pipeline()
    input = convert_channel([gold_channel_x, gold_channel_y, gold_channel_z])
    pipeline.set_input(input)
    assert True


def test_d_set_input_from_bio_data(gold_bio_data):
    pipeline = Bio_Pipeline()
    pipeline.set_input(gold_bio_data)
    assert True


def test_f_set_input_from_dict(sample_data):
    pipeline = Bio_Pipeline()
    input = convert_channel(sample_data, sampling_rate=32)
    pipeline.set_input(input)
    assert True


def test_g_set_input_from_array(sample_array_x):
    pipeline = Bio_Pipeline()
    pipeline.set_input(sample_array_x, 32, "accx")
    assert True


def test_h_add_channel(gold_channel_x):
    pipeline = Bio_Pipeline()
    pipeline.set_input(gold_channel_x)
    pipeline.input.add_channel(gold_channel_x)
    assert True


def test_j_add_channel_from_dict(sample_data):
    pipeline = Bio_Pipeline()
    input = convert_channel(sample_data, sampling_rate=32)
    pipeline.set_input(input)
    pipeline.input.add_channel(sample_data["accy"], sampling_rate=32, channel_name="accy")
    pipeline.input.add_channel(sample_data["accz"], sampling_rate=32, channel_name="accz")
    assert True


def test_k_create_process():
    mean_process = Bio_Process(np.mean, process_name="mean")
    assert True


def test_l_add_process():
    mean_process = Bio_Process(np.mean, process_name="mean")
    pipeline = Bio_Pipeline()
    pipeline.process_queue.add_process(mean_process, input_signals=["accx"], output_signals=["accx_mean"])
    assert True


def test_m_run_process(gold_channel_x):
    normalize = Bio_Process(normalize_signal, process_name="normalize")
    pipeline = Bio_Pipeline()
    pipeline.set_input(gold_channel_x)
    pipeline.process_queue.add_process(
        normalize, input_signals=["accx"], output_signals=["accx_normalized"], sampling_rate=32
    )
    pipeline.run_pipeline()
    assert True


def test_n_run_process_windowed(gold_channel_x):
    normalize = Bio_Process(normalize_signal, process_name="normalize")
    pipeline = Bio_Pipeline(windowed_process=True, window_size=10, step_size=10)
    pipeline.set_input(gold_channel_x)
    pipeline.process_queue.add_process(
        normalize, input_signals=["accx"], output_signals=["accx_normalized"], sampling_rate=32
    )
    pipeline.run_pipeline()
    assert True


def test_nn_multiple_process(gold_channel_x):

    normalize = Bio_Process(normalize_signal, process_name="normalize")
    resample = Bio_Process(resample_signal, process_name="resample")
    pipeline = Bio_Pipeline()
    pipeline.set_input(gold_channel_x)
    pipeline.process_queue.add_process(
        normalize, input_signals=["accx"], output_signals=["accx_normalized"], sampling_rate=32
    )
    pipeline.process_queue.add_process(
        resample,
        input_signals=["accx_normalized"],
        output_signals=["accx_peaks"],
        sampling_rate=32,
        target_sampling_rate=10,
        new_sr=10,
    )
    pipeline.run_pipeline()
    assert True


def test_o_multiple_process_windowed(gold_channel_x):
    normalize = Bio_Process(normalize_signal, process_name="normalize")
    resample = Bio_Process(resample_signal, process_name="resample")
    pipeline = Bio_Pipeline(windowed_process=True, window_size=10, step_size=10)
    pipeline.set_input(gold_channel_x)
    pipeline.process_queue.add_process(
        normalize, input_signals=["accx"], output_signals=["accx_normalized"], sampling_rate=32
    )
    pipeline.process_queue.add_process(
        resample,
        input_signals=["accx_normalized"],
        output_signals=["accx_resampled"],
        sampling_rate=32,
        target_sampling_rate=10,
        new_sr=10,
    )
    pipeline.run_pipeline()
    assert (
        pipeline.data["accx_resampled"].sampling_rate == 10 and pipeline.data["accx_resampled"].channel.shape[1] == 100
    )
    assert True
