import numpy as np
import pandas as pd
import pytest

from biobss.ecgtools.ecg_peaks import ecg_detectpeaks
from biobss.edatools.eda_decompose import eda_decompose
from biobss.edatools.eda_filter import filter_eda
from biobss.pipeline.bio_channel import Channel
from biobss.pipeline.bio_data import Bio_Data
from biobss.pipeline.bio_process import Bio_Process
from biobss.pipeline.channel_input import *
from biobss.pipeline.pipeline import Bio_Pipeline
from biobss.preprocess.signal_normalize import normalize_signal


def test_a_creation():
    pipeline = Bio_Pipeline()
    assert True


def test_b_creation_windowed():
    pipeline = Bio_Pipeline(windowed_process=True, window_size=10, step_size=5)
    assert True


def test_c_set_input_from_channel(ref_ecg_channel):
    pipeline = Bio_Pipeline()
    pipeline.set_input(ref_ecg_channel)
    assert True


def test_d_set_input_from_bio_data(ref_bio_data):
    pipeline = Bio_Pipeline()
    pipeline.set_input(ref_bio_data)
    assert True


def test_f_set_input_from_dict(sample_ecg_dict):
    pipeline = Bio_Pipeline()
    pipeline.set_input(sample_ecg_dict, 256, name="ecg")
    assert True


def test_g_set_input_from_array(sample_ecg_array):
    pipeline = Bio_Pipeline()
    pipeline.set_input(sample_ecg_array, 256, "ecg")
    assert True


def test_h_add_channel(ref_ecg_channel):
    pipeline = Bio_Pipeline()
    pipeline.set_input(ref_ecg_channel)
    pipeline.input.add_channel(ref_ecg_channel)
    assert True


def test_j_add_channel_from_dict(sample_ecg_array):
    pipeline = Bio_Pipeline()
    pipeline.set_input(sample_ecg_array, 256, name="ecg")
    pipeline.input.add_channel(sample_ecg_array, sampling_rate=256, channel_name="ecg")
    assert True


def test_k_create_process():
    mean_process = Bio_Process(np.mean, process_name="mean")
    assert True


def test_l_add_process():
    mean_process = Bio_Process(np.mean, process_name="mean")
    pipeline = Bio_Pipeline()
    pipeline.process_queue.add_process(mean_process, input_signals=["ecg"], output_signals=["ecg_mean"])


def test_m_run_process(ref_ecg_channel):
    normalize = Bio_Process(normalize_signal, process_name="normalize")
    pipeline = Bio_Pipeline()
    pipeline.set_input(ref_ecg_channel)
    pipeline.process_queue.add_process(
        normalize, input_signals=["ecg"], output_signals=["ecg_normalized"], sampling_rate=256
    )
    pipeline.run_pipeline()
    assert True


def test_n_run_process_windowed(ref_ecg_channel):
    normalize = Bio_Process(normalize_signal, process_name="normalize")
    pipeline = Bio_Pipeline(windowed_process=True, window_size=10, step_size=5)
    pipeline.set_input(ref_ecg_channel)
    pipeline.process_queue.add_process(
        normalize, input_signals=["ecg"], output_signals=["ecg_normalized"], sampling_rate=256
    )
    pipeline.run_pipeline()
    assert True


def test_m_multiple_process(ref_ecg_channel):

    normalize = Bio_Process(normalize_signal, process_name="normalize")
    find_peaks = Bio_Process(ecg_detectpeaks, process_name="find_peaks")
    pipeline = Bio_Pipeline()
    pipeline.set_input(ref_ecg_channel)
    pipeline.process_queue.add_process(
        normalize, input_signals=["ecg"], output_signals=["ecg_normalized"], sampling_rate=256
    )
    pipeline.process_queue.add_process(
        find_peaks, input_signals=["ecg_normalized"], output_signals=["ecg_peaks"], sampling_rate=256
    )
    pipeline.run_pipeline()
    assert True


def test_o_multiple_process_windowed(ref_ecg_channel):
    normalize = Bio_Process(normalize_signal, process_name="normalize")
    find_peaks = Bio_Process(ecg_detectpeaks, process_name="find_peaks")
    pipeline = Bio_Pipeline(windowed_process=True, window_size=10, step_size=5)
    pipeline.set_input(ref_ecg_channel)
    pipeline.process_queue.add_process(
        normalize, input_signals=["ecg"], output_signals=["ecg_normalized"], sampling_rate=256
    )
    pipeline.process_queue.add_process(
        find_peaks, input_signals=["ecg_normalized"], output_signals=["ecg_peaks"], sampling_rate=256, is_event=True
    )
    pipeline.run_pipeline()
    assert True


def test_p_multiple_process_eda(sample_eda_channel):
    pipeline = Bio_Pipeline()
    pipeline.set_input(sample_eda_channel)
    normalize = Bio_Process(normalize_signal, process_name="normalize")
    clean = Bio_Process(process_name="clean", process_method=filter_eda, method="neurokit")
    decompose = Bio_Process(process_name="decompose", process_method=eda_decompose, method="highpass")
    pipeline.process_queue.add_process(
        normalize, input_signals=["eda_raw"], output_signals=["eda_normalized"], sampling_rate=700
    )
    pipeline.process_queue.add_process(
        clean, input_signals=["eda_normalized"], output_signals=["eda_clean"], sampling_rate=700
    )
    pipeline.process_queue.add_process(
        decompose, input_signals=["eda_clean"], output_signals=["eda_tonic", "eda_phasic"], sampling_rate=700
    )
    pipeline.run_pipeline()
    assert True


def test_q_multiple_process_eda_windowed(sample_eda_channel):
    pipeline = Bio_Pipeline(windowed_process=True, window_size=100, step_size=50)
    pipeline.set_input(sample_eda_channel)
    normalize = Bio_Process(normalize_signal, process_name="normalize")
    clean = Bio_Process(process_name="clean", process_method=filter_eda, method="neurokit")
    decompose = Bio_Process(process_name="decompose", process_method=eda_decompose, method="highpass")
    pipeline.process_queue.add_process(
        normalize, input_signals=["eda_raw"], output_signals=["eda_normalized"], sampling_rate=700
    )
    pipeline.process_queue.add_process(
        clean, input_signals=["eda_normalized"], output_signals=["eda_clean"], sampling_rate=700
    )
    pipeline.process_queue.add_process(
        decompose, input_signals=["eda_clean"], output_signals=["eda_tonic", "eda_phasic"], sampling_rate=700
    )
    pipeline.run_pipeline()

    assert True
