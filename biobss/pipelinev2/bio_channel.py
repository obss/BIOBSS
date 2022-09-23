import numpy as np
import copy as copy
from numpy.typing import ArrayLike


class Bio_Channel():
    """ Signal object with add and iterate process objects"""

    def __init__(self, signal: ArrayLike,name:str, sampling_rate: float, timestamp=None, timestamp_resolution='ms', timestamp_start=0):

        regularity_parameter = 0.01
        if(timestamp_resolution == 'ns'):
            regularity_parameter = 1e-9
        elif(timestamp_resolution == 'ms'):
            regularity_parameter = 0.001
        elif(timestamp_resolution == 's'):
            regularity_parameter = 1
        elif(timestamp_resolution == 'min'):
            regularity_parameter = 60
        else:
            raise ValueError('timestamp_resolution must be "ns","ms","s","min"')
            
            
        self.channel = np.array(signal)
        
        if(sampling_rate < 0):
            raise ValueError('Sampling rate must be greater than 0')

        self.sampling_rate = sampling_rate
        if(timestamp is not None):
            timestamp = np.array(timestamp)
            if(timestamp.shape != signal.shape):
                raise ValueError(timestamp.shape, signal.shape,
                                 'Timestamp must match signal dimensions')
            if(np.any(np.diff(timestamp) < 0)):
                raise ValueError('Timestamp must be monotonic')
            if(np.diff(timestamp).std() > regularity_parameter):  # TODO: optimize this parameter
                raise ValueError('Timestamp must be regularly spaced')
            self.timestamp = timestamp
            self.timestamp_start = timestamp[0]
            print("Input is set with timestamp resolution of " + str(timestamp_resolution))
        else:
            if(timestamp_start < 0):
                raise ValueError('Timestamp start must be greater than 0')
            timestamp_factor=1
            if(timestamp_resolution == 'ns'):
                timestamp_factor = 1/1e-9
            elif(timestamp_resolution == 'ms'):
                timestamp_factor = 1/0.001
            elif(timestamp_resolution == 's'):
                timestamp_factor = 1
            elif(timestamp_resolution == 'min'):
                timestamp_factor = 60
                
            timestamp = (np.arange(len(signal))/sampling_rate)*timestamp_factor
            self.timestamp = timestamp+timestamp_start

        self.timestamp_start = timestamp_start
        self.signal_name = name
        if(len(signal.shape) < 2):
            self.signal_duration = len(signal)/sampling_rate,
            self.windows = 1
        else:
            self.signal_duration = signal.shape[1]/sampling_rate
            self.windows = signal.shape[0]

    def copy(self):
        return copy.deepcopy(self)

    def sync_timestamps(self, offset: float):
        if(len(self.timestamp.shape) == 1):
            self.timestamp = self.timestamp+offset
        else:
            self.timestamp[:, 0] = self.timestamp[:, 0]+offset

    def change_channel_data(self, signal: ArrayLike):
        self.channel = np.array(signal)

    def get_timestamp(self, ts_point="start") -> np.ndarray:

        if(not ts_point in ["start", "end", "mid"]):
            raise ValueError('ts_point must be "start","end","mid"')

        if(len(self.timestamp.shape) == 1):
            out = np.array(self.timestamp)
        else:
            if(ts_point == "start"):
                out = np.array(self.timestamp[:, 0])
            elif(ts_point == "end"):
                out = np.array(self.timestamp[:, -1])
            elif(ts_point == "mid"):
                out = np.array(
                    self.timestamp[:, int(len(self.timestamp[0])/2)])

        return out

    def __str__(self) -> str:
        return str(self.channel)

    def __repr__(self) -> str:
        representation = self.signal_name
        representation += " (" + self.signal_modality + ")"
        representation += " (" + str(self.sampling_rate) + "Hz)"
        representation += " (" + str(self.signal_duration) + "s)"
        representation += " (" + str(self.windows) + " windows)"
        representation += " (" + str(self.channel.shape) + ")"
        representation += " (" + str(self.timestamp.shape) + ")"
        representation += " (" + str(self.channel.dtype) + ")"
        representation += self.__str__()
        return representation
