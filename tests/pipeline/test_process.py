from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from biobss.pipeline.bio_channel import Channel
from biobss.pipeline.bio_data import Bio_Data
from biobss.pipeline.bio_process import Bio_Process
from biobss.pipeline.bioprocess_queue import Process_List
from biobss.pipeline.channel_input import *
from biobss.preprocess.signal_normalize import normalize_signal
from biobss.preprocess.signal_segment import segment_signal


@pytest.fixture()
def gold_channel(sample_ecg_array):
    test_data = np.array(sample_ecg_array)
    test_data = test_data.flatten()
    gold_channel_ = Channel(test_data, name="ecg", sampling_rate=256)
    return gold_channel_


@pytest.fixture()
def gold_bio_data(gold_channel):
    gold_bio_data_ = Bio_Data()
    gold_bio_data_.add_channel(gold_channel)
    return gold_bio_data_


@pytest.fixture()
def sample_windowed(sample_ecg_array):
    test_data = sample_ecg_array
    test_data = segment_signal(test_data, sampling_rate=256, window_size=10, step_size=5)
    return test_data


def test_a_process_creation():
    normalize = Bio_Process(normalize_signal, process_name="normalize")
    assert True


def test_b_process_run(sample_ecg_array):
    normalize = Bio_Process(normalize_signal, process_name="normalize")
    norm = normalize.run(sample_ecg_array)
    assert np.all(norm == normalize_signal(sample_ecg_array))


def test_c_process_run_windowed(sample_windowed):
    normalize = Bio_Process(normalize_signal, process_name="normalize")
    norm = normalize.run(sample_windowed)
    assert np.all(norm == normalize_signal(sample_windowed))


def test_d_process_queue(gold_bio_data):
    normalize = Bio_Process(normalize_signal, process_name="normalize")
    test_queue = Process_List()
    test_queue.add_process(normalize, input_signals=["ecg"], output_signals=["ecg_norm"], sampling_rate=256)
    test_queue.run_process_queue(gold_bio_data)
    assert True


def test_f_queue_results(sample_ecg_array, gold_bio_data):
    normalize = Bio_Process(normalize_signal, process_name="normalize")
    test_queue = Process_List()
    test_queue.add_process(normalize, input_signals=["ecg"], output_signals=["ecg_norm"], sampling_rate=256)
    results = test_queue.run_process_queue(gold_bio_data)
    test_channel = Channel(normalize_signal(sample_ecg_array), name="ecg_norm", sampling_rate=256)
    test_data = deepcopy(gold_bio_data)
    test_data.add_channel(test_channel)
    assert test_data == results
