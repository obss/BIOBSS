from __future__ import annotations

from copy import copy
from typing import Union

import pandas as pd
from numpy.typing import ArrayLike

from ..preprocess.signal_segment import segment_signal
from .bio_channel import Channel
from .bio_data import Bio_Data

# from .. import preprocess
from .bioprocess_queue import Process_List
from .channel_input import *
from .event_input import *
from .feature_extraction import Feature
from .feature_queue import Feature_Queue

"""a biological signal processing object with preprocessing and postprocessing steps"""


class Bio_Pipeline:
    def __init__(
        self,
        windowed_process=False,
        window_size=None,
        step_size=None,
    ):
        if windowed_process:
            self.windowed = True
            if window_size is None or step_size is None:
                raise ValueError("window_size and step_size must be specified")
            else:
                self.set_window_parameters(window_size, step_size)
        else:
            self.window_size = "Not Windowed"
            self.step_size = "Not Windowed"
            self.windowed = False

        self.process_queue = Process_List(name="Process_List")
        self.features = pd.DataFrame()
        self.feature_list = Feature_Queue()

    def set_input(
        self,
        data: Union[Bio_Data, ArrayLike, Channel],
        sampling_rate=None,
        name=None,
        is_event=False,
        **kwargs,
    ):

        if isinstance(data, Bio_Data):
            self.input = data
        elif isinstance(data, Channel):
            self.input = Bio_Data()
            self.input.add_channel(data)
        else:
            if sampling_rate is None:
                raise ValueError("If signal is not a Bio_Data or Bio_Channel object, sampling_rate must be specified")
            if name is None and not isinstance(data, dict):
                raise ValueError("If signal is not a Bio_Data or Bio_Channel object, name must be specified")
            if is_event:
                self.input = convert_event(data, sampling_rate, name, **kwargs)
            else:
                self.input = convert_channel(data, sampling_rate, name, **kwargs)

    def set_window_parameters(self, window_size=10, step_size=5):
        self.window_size = window_size
        self.step_size = step_size

    def convert_windows(self):
        for ch in self.data.get_channel_names():
            channel = self.data[ch]
            if isinstance(channel, Event_Channel):
                is_event = True
            else:
                is_event = False

            windowed = segment_signal(
                signal=channel.channel,
                window_size=self.window_size,
                step_size=self.step_size,
                sampling_rate=channel.sampling_rate,
                is_event=is_event,
            )
            if is_event:
                windowed = Event_Channel(windowed, name=channel.signal_name, sampling_rate=channel.sampling_rate)
            else:
                windowed = Channel(windowed, name=channel.signal_name, sampling_rate=channel.sampling_rate)

        self.data[ch] = windowed
        self.feature_list.windowed = True
        self.segmented = True
        pass

    def extract_features(self):
        self.features = self.feature_list.run_feature_queue(self.data)

    def add_feature_step(self, feature: Feature, input_signals, *args, **kwargs):
        self.feature_list.add_feature(feature, input_signals, *args, **kwargs)

    def run_pipeline(self):
        try:
            self.data = copy(self.input)
        except AttributeError:
            raise ValueError("Input data must be set before running pipeline")

        # self.data = self.preprocess_queue.run_process_queue(self.data)
        if self.windowed:
            self.convert_windows()
        self.data = self.process_queue.run_process_queue(self.data)

    def get_features(self):
        return copy(self.features)

    def get_data(self):
        return copy(self.data)

    def get_input(self):
        return copy(self.input)

    def export_data(self, filename):
        self.data.export_data(filename)

    def export_features(self, filename):
        self.features.to_csv(filename)

    def clear_data(self):
        self.data = None

    def clear_features(self):
        self.features = None

    def clear_input(self):
        self.input = None

    def __repr__(self) -> str:
        representation = "Bio_Pipeline:\n"
        representation += "\tProcessors: " + str(self.process_queue) + "\n"
        representation += "\tWindow Size(Seconds): " + str(self.window_size) + "\n"
        representation += "\tStep Size: " + str(self.step_size) + "\n"

        return representation
