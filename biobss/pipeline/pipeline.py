import time
from .. import signaltools
from .bioprocess_queue import Process_List
from .bio_data import Bio_Data
from .bio_channel import Bio_Channel
from typing import Union
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from .feature_extraction import Feature

"""a biological signal processing object with preprocessing and postprocessing steps"""


class Bio_Pipeline:

    def __init__(
        self,
        windowed_process=False,
        window_size=None,
        step_size=None,
        features_list=[],
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

        self.preprocess_queue = Process_List(name="Preprocess_Queue")
        self.process_queue = Process_List(name="Process_Queue")
        self.postprocess_queue = Process_List(name="Postprocess_Queue")
        self.features = pd.DataFrame()
        self.feature_list = features_list

    def set_input(
        self,
        signal: Union[Bio_Data, ArrayLike],
        sampling_rate=None,
        name=None,
        timestamp=None,
        timestamp_start=0,
        timestamp_resolution='ms',
    ):

        if isinstance(signal, Bio_Data):
            self.input = signal
        else:
            if sampling_rate is None:
                raise ValueError(
                    "If signal is not a Bio_Data object, sampling_rate must be specified"
                )
            if name is None:
                raise ValueError(
                    "If signal is not a Bio_Data object, name must be specified"
                )
            self.input = Bio_Data()
            if isinstance(signal, (np.ndarray, pd.Series, list)):
                self.input.add_channel(
                    Bio_Channel(
                        np.array(signal),
                        sampling_rate=sampling_rate,
                        name=name,
                        timestamp=timestamp,
                        timestamp_start=timestamp_start,
                        timestamp_resolution=timestamp_resolution,
                    )
                )
            elif isinstance(signal, pd.DataFrame):
                for column in signal.columns:
                    self.input.add_channel(
                        Bio_Channel(
                            signal[column],
                            sampling_rate=sampling_rate,
                            name=column,
                            timestamp=signal.index,
                            timestamp_start=timestamp_start,
                            timestamp_resolution=timestamp_resolution,
                        )
                    )
            else:
                raise ValueError(
                    "Input signal must be a Bio_Data object, a pandas DataFrame, a pandas Series, a numpy array, or a list"
                )

    def set_window_parameters(self, window_size=10, step_size=5):
        self.window_size = window_size
        self.step_size = step_size

    def convert_windows(self):
        for ch in self.input.get_channel_names():
            channel = self.input[ch]
            windowed = signaltools.segment_signal(
                channel.channel,
                self.window_size,
                self.step_size,
                sampling_rate=channel.sampling_rate,
            )
            timestamps = signaltools.segment_signal(
                channel.timestamp,
                self.window_size,
                self.step_size,
                sampling_rate=channel.sampling_rate,
            )
            self.input.modify_signal(
                windowed,
                channel_name=channel.signal_name,
                timestamp=timestamps,
                sampling_rate=channel.sampling_rate,
                timestamp_resolution=channel.timestamp_resolution
            )

        self.segmented = True

    def extract_features(self):
        for f in self.feature_list:
            self.calculate_feature(f)

    def add_feature_step(self, feature: Feature):
        self.feature_list.append(feature)

    def calculate_feature(self, feature: Feature):
        self.features = pd.concat(
            [self.features, feature.run(self.input)], axis=1)

    def run_pipeline(self):

        self.input = self.preprocess_queue.run_process_queue(self.input)
        if self.windowed:
            self.convert_windows()

        self.input = self.process_queue.run_process_queue(self.input)

    def __repr__(self) -> str:
        representation = "Bio_Pipeline:\n"
        representation += "\tPreprocessors: " + \
            str(self.preprocess_queue) + "\n"
        representation += "\tProcessors: " + str(self.process_queue) + "\n"
        representation += "\tPostprocessors: " + \
            str(self.postprocess_queue) + "\n"
        representation += "\tWindow Size(Seconds): " + \
            str(self.window_size) + "\n"
        representation += "\tStep Size: " + str(self.step_size) + "\n"

        return representation
