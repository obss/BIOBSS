import numpy as np
import pytest

from biobss.preprocess.signal_segment import *
from biobss.utils.sample_loader import *


def test_segment_length(load_sample_eda):

    data, info = load_sample_eda

    sig = np.asarray(data["EDA"])
    fs = info["sampling_rate"]
    L = info["signal_length"]

    segment_length = 600
    step_size = 100

    num_frames = int(np.floor((L * fs - int(segment_length * fs)) / int(segment_length * fs)) + 1)
    num_frames_sliding = int(np.floor((L * fs - int(segment_length * fs)) / int(step_size * fs)) + 1)

    segmented = segment_signal(sig, window_size=segment_length, step_size=segment_length, sampling_rate=fs)
    segmented_sliding = segment_signal(sig, window_size=segment_length, step_size=step_size, sampling_rate=fs)

    assert np.shape(segmented)[0] == num_frames
    assert np.shape(segmented)[1] == int(segment_length * fs)
    assert np.shape(segmented_sliding)[0] == num_frames_sliding
    assert np.shape(segmented_sliding)[1] == int(segment_length * fs)
