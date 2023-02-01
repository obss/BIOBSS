import sys

import numpy as np
import pandas as pd
import pytest

from biobss.pipeline.bio_channel import Channel
from biobss.pipeline.bio_data import Bio_Data
from biobss.pipeline.bio_process import Bio_Process
from biobss.pipeline.channel_input import *
from biobss.pipeline.feature_extraction import Feature
from biobss.pipeline.pipeline import Bio_Pipeline
from biobss.ppgtools.ppg_features import *
from biobss.ppgtools.ppg_freqdomain import *
from biobss.ppgtools.ppg_statistical import *
from biobss.ppgtools.ppg_timedomain import *
from biobss.preprocess.signal_normalize import normalize_signal


@pytest.fixture()
def gold_channel(load_sample_ppg):
    test_data, info = load_sample_ppg
    sampling_rate = info["sampling_rate"]
    test_data = np.array(test_data.values).flatten()
    gold_channel_ = Channel(test_data, name="ppg", sampling_rate=sampling_rate)
    return gold_channel_


@pytest.fixture()
def gold_bio_data(load_sample_ppg):
    test_data, info = load_sample_ppg
    sampling_rate = info["sampling_rate"]
    test_data = np.array(test_data.values).flatten()
    gold_channel_ = Channel(test_data, name="ppg", sampling_rate=sampling_rate)
    gold_bio_data_ = Bio_Data()
    gold_bio_data_.add_channel(gold_channel_)
    return gold_bio_data_


@pytest.fixture()
def sample_data(load_sample_ppg):
    test_data, info = load_sample_ppg
    sampling_rate = info["sampling_rate"]
    test_data = np.array(test_data.values).flatten()
    sample_data_ = {"ppg": test_data}
    return sample_data_


@pytest.fixture()
def sample_array(load_sample_ppg):
    test_data, info = load_sample_ppg
    sampling_rate = info["sampling_rate"]
    test_data = np.array(test_data.values).flatten()
    return test_data


def test_a_creation():
    pipeline = Bio_Pipeline()
    assert True


def test_b_creation_windowed():
    pipeline = Bio_Pipeline(windowed_process=True, window_size=1, step_size=1)
    assert True


def test_c_set_input_from_channel(gold_channel):
    pipeline = Bio_Pipeline()
    pipeline.set_input(gold_channel)
    assert True


def test_d_set_input_from_bio_data(gold_bio_data):
    pipeline = Bio_Pipeline()
    pipeline.set_input(gold_bio_data)
    assert True


def test_f_set_input_from_dict(sample_data):
    pipeline = Bio_Pipeline()
    pipeline.set_input(sample_data, 64)
    assert True


def test_g_set_input_from_array(sample_array):
    pipeline = Bio_Pipeline()
    pipeline.set_input(sample_array, 64, "ppg")
    assert True


def test_h_add_channel(gold_channel):
    pipeline = Bio_Pipeline()
    pipeline.set_input(gold_channel)
    pipeline.input.add_channel(gold_channel)
    assert True


def test_j_add_channel_from_dict(sample_data):
    pipeline = Bio_Pipeline()
    pipeline.set_input(sample_data, 64, name="ppg")
    pipeline.input.add_channel(sample_data, sampling_rate=64, channel_name="ppg")
    assert True


def test_k_create_process():
    mean_process = Bio_Process(np.mean, process_name="mean")
    assert True


def test_l_add_process():
    mean_process = Bio_Process(np.mean, process_name="mean")
    pipeline = Bio_Pipeline()
    pipeline.process_queue.add_process(mean_process, input_signals=["ppg"], output_signals=["ppg_mean"])


def test_m_run_process(gold_channel):
    normalize = Bio_Process(normalize_signal, process_name="normalize")
    pipeline = Bio_Pipeline()
    pipeline.set_input(gold_channel)
    pipeline.process_queue.add_process(
        normalize, input_signals=["ppg"], output_signals=["ppg_normalized"], sampling_rate=64
    )
    pipeline.run_pipeline()
    assert True


def test_n_run_process_windowed(gold_channel):
    normalize = Bio_Process(normalize_signal, process_name="normalize")
    pipeline = Bio_Pipeline(windowed_process=True, window_size=10, step_size=1)
    pipeline.set_input(gold_channel)
    pipeline.process_queue.add_process(
        normalize, input_signals=["ppg"], output_signals=["ppg_normalized"], sampling_rate=64
    )
    pipeline.run_pipeline()
    assert True


def test_m_multiple_process(gold_channel):

    normalize = Bio_Process(normalize_signal, process_name="normalize")
    find_peaks = Bio_Process(ppg_detectpeaks, process_name="find_peaks")
    pipeline = Bio_Pipeline()
    pipeline.set_input(gold_channel)
    pipeline.process_queue.add_process(
        normalize, input_signals=["ppg"], output_signals=["ppg_normalized"], sampling_rate=64
    )
    pipeline.process_queue.add_process(
        find_peaks, input_signals=["ppg_normalized"], output_signals=["ppg_peaks"], sampling_rate=64, delta=0.01
    )
    pipeline.run_pipeline()
    assert True


def test_o_multiple_process_windowed(gold_channel):
    normalize = Bio_Process(normalize_signal, process_name="normalize")
    find_peaks = Bio_Process(ppg_detectpeaks, process_name="find_peaks", return_index="Peak_locs")
    find_onsets = Bio_Process(ppg_detectpeaks, process_name="find_onsets", return_index="Trough_locs")
    pipeline = Bio_Pipeline(windowed_process=True, window_size=10, step_size=1)
    pipeline.set_input(gold_channel)
    pipeline.process_queue.add_process(
        normalize, input_signals=["ppg"], output_signals=["ppg_normalized"], sampling_rate=64
    )
    pipeline.process_queue.add_process(
        find_peaks, input_signals=["ppg_normalized"], output_signals=["ppg_peaks"], sampling_rate=64, delta=0.01
    )
    pipeline.process_queue.add_process(
        find_onsets, input_signals=["ppg_normalized"], output_signals=["ppg_onsets"], sampling_rate=64, delta=0.01
    )
    pipeline.run_pipeline()
    assert True


def test_p_feature_process(gold_channel):
    normalize = Bio_Process(normalize_signal, process_name="normalize")
    find_peaks = Bio_Process(ppg_detectpeaks, process_name="find_peaks", return_index="Peak_locs")
    correct_peaks = Bio_Process(peak_control, process_name="correct_peaks")
    pipeline = Bio_Pipeline(windowed_process=True, window_size=10, step_size=1)
    pipeline.set_input(gold_channel)
    pipeline.process_queue.add_process(
        normalize, input_signals=["ppg"], output_signals=["ppg_normalized"], sampling_rate=64
    )
    pipeline.process_queue.add_process(
        find_peaks,
        input_signals=["ppg_normalized"],
        output_signals=["ppg_peaks", "ppg_onsets"],
        sampling_rate=64,
        delta=0.01,
        is_event=True,
    )
    pipeline.process_queue.add_process(
        correct_peaks,
        input_signals=["ppg_normalized", "ppg_peaks", "ppg_onsets"],
        output_signals=["corrected_peaks", "corrected_onsets"],
        sampling_rate=64,
        is_event=True,
    )

    pipeline.run_pipeline()

    segment_features = Feature(
        name="segment_features", function=from_segment, feature_types=["Freq", "Time", "Stat"], sampling_rate=64
    )
    cycle_features = Feature(
        name="cycle_features", function=from_cycles, feature_types=["Time", "Stat"], sampling_rate=64
    )
    pipeline.add_feature_step(segment_features, feature_prefix="ppg_segment", input_signals=["ppg_normalized"])
    pipeline.add_feature_step(
        cycle_features,
        feature_prefix="ppg_cycle",
        input_signals={"sig": "ppg_normalized", "peaks_locs": "corrected_peaks", "troughs_locs": "corrected_onsets"},
    )

    pipeline.extract_features()

    assert True
