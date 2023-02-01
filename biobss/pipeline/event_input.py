import numpy as np
import pandas as pd

from .bio_data import Bio_Data
from .event_channel import Event_Channel


def event_from_signal(signal, indicator, n_windows=1):

    if not isinstance(signal, (np.ndarray, list)):
        raise ValueError("signal must be a list or numpy array")

    if n_windows == 1:
        signal = np.array(signal)
        event = np.where(signal == indicator)[0]
    elif n_windows > 1:
        event = []
        for i in range(n_windows):
            event.append(np.where(signal[i] == indicator)[0])
    return event


def convert_event(event, name=None, sampling_rate=None, indicator=None, n_windows=1, n_signal=None):

    if isinstance(event, Event_Channel):
        output = Bio_Data()
        output.add_channel(event)
        return output

    if indicator is not None:
        if not isinstance(indicator, (int, float)):
            raise ValueError("indicator must be a number")
        elif not isinstance(event, (np.ndarray, list)):
            raise ValueError("event must be a list or numpy array")
        event = event_from_signal(event, indicator, n_windows)

    output = Bio_Data()
    if isinstance(event, (np.ndarray, list)):
        if name is None:
            name = "event"
        result = _from_array_list(event, name, sampling_rate, n_windows, n_signal)
        if isinstance(result, (Event_Channel)):
            output.add_channel(result)
        elif isinstance(result, Bio_Data):
            output = result
        else:
            raise ValueError("result must be an Event_Channel or Bio_Data object")
        return output

    if isinstance(event, (dict)):
        result = _from_dict(event, name, sampling_rate, n_windows, n_signal)
        return result

    elif isinstance(event, pd.DataFrame):
        result = _from_dataframe(event, name, sampling_rate, n_windows, n_signal)
        return result

    pass


def _from_dataframe(event, name, sampling_rate, n_windows=1, n_signal=None):

    if not isinstance(event, pd.DataFrame):
        raise ValueError("event must be a pandas DataFrame")
    output = Bio_Data()
    if isinstance(name, (list)):
        if len(name) == 1:
            name = name[0]
        elif len(name) != len(event.columns):
            raise ValueError("name must be a list with the same length as the number of columns in event")
    if isinstance(name, str):
        if len(event.columns) != 1:
            modified = {}
            for k, v in event.items():
                mod_name = name + "_" + k
                modified[mod_name] = v
            event = modified

    for key in event.keys():
        if isinstance(name, (list)):
            output.add_channel(_from_array_list(event[key].tolist(), name.pop(0), sampling_rate, n_windows, n_signal))
        else:
            output.add_channel(_from_array_list(event[key].tolist(), key, sampling_rate, n_windows, n_signal))
    return output


def _from_dict(event, name, sampling_rate, n_windows=1, n_signal=None):

    if not isinstance(event, dict):
        raise ValueError("event must be a dictionary")
    output = Bio_Data()
    if isinstance(name, (list)):
        if len(name) == 1:
            name = name[0]
        elif len(name) != len(event.keys()):
            raise ValueError("name must be a list with the same length as the number of keys in event")
    if isinstance(name, str):
        modified = {}
        for k, v in event.items():
            mod_name = name + "_" + k
            modified[mod_name] = v
        event = modified

    for key in event.keys():
        if isinstance(name, (list)):
            output.add_channel(_from_array_list(event[key], name.pop(0), sampling_rate, n_windows, n_signal))
        else:
            output.add_channel(_from_array_list(event[key], key, sampling_rate, n_windows, n_signal))

    return output


def _from_array_list(event, name, sampling_rate, n_windows=1, n_signal=None):

    if not isinstance(event, (np.ndarray, list)):
        raise ValueError("event must be a list or numpy array")
    if n_signal is None:
        n_signal = 1
    if n_windows > 1 or n_signal > 1:
        if n_signal is None:
            n_signal = 1
        if not all(isinstance(x, (np.ndarray, dict)) for x in event):
            event = np.array(event, dtype=object)
            event = event.reshape(n_windows, n_signal, -1)
        else:
            if len(event) != n_windows:
                raise ValueError("event must have the same number of windows as n_windows")

    else:
        event = [event]
    if isinstance(name, (list)):
        name = name[0]

    output = []
    if all(isinstance(x, dict) for x in event):
        event = convert_list_dict_to_dict_list(event)
        return _from_dict(event, name, sampling_rate, n_windows, n_signal)
    else:
        for i in range(n_windows):
            data = np.array(event[i], dtype=object)
            data = data.squeeze()
            output.append(data.tolist())
        if n_windows == 1:
            output = output[0]
        output = Event_Channel(output, name, sampling_rate)

        return output


def convert_list_dict_to_dict_list(list_dict):
    result = {}
    for key in list_dict[0].keys():
        result[key] = [d[key] for d in list_dict]
    return result
