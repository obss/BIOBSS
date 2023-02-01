import numpy as np
import pandas as pd

from .bio_channel import Channel
from .bio_data import Bio_Data


def convert_channel(signal, sampling_rate=None, name=None, n_windows=1, n_signal=None, index=None):

    if isinstance(signal, Channel):
        output = Bio_Data()
        output.add_channel(signal)
        return output
    elif isinstance(signal, Bio_Data):
        return signal
    elif isinstance(signal, dict):
        output = _from_dict(signal, sampling_rate, name, n_windows, n_signal, index)
    elif isinstance(signal, list):
        output = _from_list(signal, sampling_rate, name, n_windows, n_signal, index)
    elif isinstance(signal, tuple):
        output = _from_tuple(signal, sampling_rate, name, n_windows, n_signal, index)
    elif isinstance(signal, np.ndarray):
        output = _from_array(signal, sampling_rate, name, n_windows, n_signal, index)
    elif isinstance(signal, pd.DataFrame):
        output = _from_dataframe(signal, sampling_rate, name, n_windows, n_signal, index)

    return output


def _from_dataframe(signal, sampling_rate=None, name=None, n_windows=1, n_signal=None, index=None):
    if name is not None:
        if isinstance(name, str):
            if len(signal.columns) == 1:
                signal.columns = [name]
            else:
                raise ValueError("name must be a list of strings if signal has more than one column")
        elif isinstance(name, list):
            if len(signal.columns) == len(name):
                signal.columns = name
            else:
                raise ValueError("name must be a list of strings if signal has more than one column")

    if index is not None:
        if isinstance(index, str):
            signal = signal[index]
        elif isinstance(index, list):
            if any([not isinstance(key, str) for key in index]):
                raise ValueError("index must be a list of strings")
        else:
            new_signal = {}
            for key in index:
                new_signal[key] = signal[key]
            signal = new_signal

    _check_sampling_rate(sampling_rate, n_signal)
    if len(signal.keys()) == 0:
        raise ValueError("signal must have at least one key")
    output = Bio_Data()
    for key in signal.keys():
        data = signal[key]
        data = np.array(data)
        data = data.reshape(n_windows, -1)
        data = data.squeeze()
        output.add_channel(Channel(data, key, sampling_rate))

    return output


def _from_dict(signal, sampling_rate=None, name=None, n_windows=1, n_signal=None, index=None):

    if index is not None:
        if isinstance(index, str):
            signal = signal[index]
        elif isinstance(index, list):
            if any([not isinstance(key, str) for key in index]):
                raise ValueError("index must be a list of strings")
        else:
            new_signal = {}
            for key in index:
                new_signal[key] = signal[key]
            signal = new_signal

    _check_sampling_rate(sampling_rate, n_signal)
    if len(signal.keys()) == 0:
        raise ValueError("signal must have at least one key")
    output = Bio_Data()
    for key in signal.keys():
        if not isinstance(signal[key], (list, tuple, np.ndarray)):
            raise ValueError("signal must be a list, tuple or numpy array")
        data = signal[key]
        data = np.array(data)
        data = data.reshape(n_windows, -1)
        data = data.squeeze()
        output.add_channel(Channel(data, key, sampling_rate))

    return output


def _from_list(signal, sampling_rate=None, name=None, n_windows=1, n_signal=None, index=None):

    if index is not None:
        if isinstance(index, int):
            signal = [signal[index]]
        elif isinstance(index, list):
            if any([not isinstance(key, int) for key in index]):
                raise ValueError("index must be a list of integers")
        else:
            new_signal = []
            for key in index:
                new_signal.append(signal[key])
            signal = new_signal

    if all(isinstance(item, (Channel,)) for item in signal):
        output = Bio_Data()
        if name is not None:
            if isinstance(name, str):
                for i in range(len(signal)):
                    signal[i].signal_name = name + "_" + str(i)
            elif isinstance(name, list):
                for i in range(len(name)):
                    signal[i].signal_name = name[i]
        for s in signal:
            output.add_channel(s)
        return output

    _check_sampling_rate(sampling_rate, n_signal)

    if any([not isinstance(item, (list, np.ndarray)) for item in signal]):
        raise ValueError("signal must be a list of lists, tuples or numpy arrays")

    if name is None:
        name = "signal"
    elif isinstance(name, list):
        if len(name) != len(signal):
            raise ValueError("name must be a string or a list of length len(signal)")

    signal = np.array(signal)
    if n_signal is None:
        n_signal = 1

    output = Bio_Data()
    signal = signal.reshape(n_signal, n_windows, -1)

    for i in range(signal.shape[0]):
        data = signal[i]
        data = data.reshape(n_windows, -1)
        data = data.squeeze()
        if name == "signal":
            cur_name = "signal_" + str(i)
        elif isinstance(name, list):
            cur_name = name[i]
        elif isinstance(name, str):
            cur_name = name
        else:
            raise ValueError("name must be a string or a list of length len(signal)")

        if not isinstance(sampling_rate, (int, float)):
            cur_rate = sampling_rate[i]
        else:
            cur_rate = sampling_rate

        output.add_channel(Channel(data, cur_name, cur_rate))
    return output


def _from_tuple(signal, sampling_rate=None, name=None, n_windows=1, n_signal=None, index=None):
    if not isinstance(signal, tuple):
        raise ValueError("signal must be a tuple")
    if index is not None:
        if isinstance(index, int):
            signal = [signal[index]]
        elif isinstance(index, list):
            if any([not isinstance(key, int) for key in index]):
                raise ValueError("index must be a list of integers")
        else:
            new_signal = []
            for key in index:
                new_signal.append(signal[key])
            signal = new_signal

    if all(isinstance(item, (Channel,)) for item in signal):
        output = Bio_Data()
        for s in signal:
            output.add_channel(s)
        return output
    if n_signal is not None:
        if len(signal) != n_signal:
            raise ValueError("signal must be a tuple of length n_signal")
    if name is None:
        name = "signal"

    output = Bio_Data()
    _check_sampling_rate(sampling_rate, n_signal)
    for i, s in enumerate(signal):
        data = np.array(s)
        data = data.reshape(n_windows, -1)
        data = data.squeeze()
        if name == "signal":
            cur_name = "signal_" + str(i)
        elif isinstance(name, list):
            cur_name = name[i]
        else:
            cur_name = name
        if isinstance(sampling_rate, (int, float)):
            cur_rate = sampling_rate
        elif isinstance(sampling_rate, list):
            cur_rate = sampling_rate[i]

        output.add_channel(Channel(data, cur_name, cur_rate))

    return output


def _from_array(signal, sampling_rate=None, name=None, n_windows=1, n_signal=None, index=None):

    if index is not None:
        if isinstance(index, int):
            signal = signal[index]
        elif isinstance(index, list):
            if any([not isinstance(key, int) for key in index]):
                raise ValueError("index must be a list of integers")
        else:
            new_signal = []
            for key in index:
                new_signal.append(signal[key])
            signal = new_signal

    if not isinstance(signal, (np.ndarray)):
        raise ValueError("signal must be a numpy array")
    if n_signal is not None:
        if signal.shape[0] != n_signal:
            raise ValueError("signal must be a numpy array of shape (n_signal,n_windows,n_samples)")
    else:
        n_signal = 1
    if name is None:
        name = "signal"

    _check_sampling_rate(sampling_rate, n_signal)
    output = Bio_Data()
    signal = signal.reshape(n_signal, n_windows, -1)
    for i in range(signal.shape[0]):
        data = signal[i]
        data = data.reshape(n_windows, -1)
        data = data.squeeze()
        if name == "signal":
            cur_name = "signal_" + str(i)
        elif isinstance(name, list):
            cur_name = name[i]
        else:
            cur_name = name
        if isinstance(sampling_rate, (int, float)):
            cur_rate = sampling_rate
        elif isinstance(sampling_rate, list):
            cur_rate = sampling_rate[i]

        output.add_channel(Channel(data, cur_name, cur_rate))

    return output


def _check_sampling_rate(sampling_rate, n_signal):
    if sampling_rate is None:
        raise ValueError("sampling_rate must be provided")
    elif not isinstance(sampling_rate, (int, float, list)):
        raise ValueError("sampling_rate must be a number or a list of length n_signal")
    elif isinstance(sampling_rate, list):
        if any([not isinstance(rate, (int, float)) for rate in sampling_rate]):
            raise ValueError("sampling_rate must be a number or a list of length n_signal")
    if n_signal is not None:
        if isinstance(sampling_rate, (int, float)):
            sampling_rate = sampling_rate
        elif isinstance(sampling_rate, list):
            if len(sampling_rate) != n_signal:
                raise ValueError("sampling_rate must be a number or a list of length n_signal")
            for rate in sampling_rate:
                if not isinstance(rate, (int, float)):
                    raise ValueError("sampling_rate must be a number or a list of length n_signal")
                elif rate <= 0:
                    raise ValueError("sampling_rate must be positive")
    return True
