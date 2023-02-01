from __future__ import annotations

from .bio_data import Bio_Data
from .channel_input import *
from .event_input import *


class Process_List:
    def __init__(self, name="Process_Queue"):
        self.process_list = []
        self.input_signals = []
        self.output_signals = []
        self.kwargs = []
        self.args = []
        self.is_event = []
        self.name = name
        self.processed_index = 0

    def add_process(self, process, input_signals=None, output_signals=None, is_event=False, *args, **kwargs):
        self.process_list.append(process)
        self._process_io(input_signals, output_signals)
        self.is_event.append(is_event)
        if isinstance(kwargs, dict):
            self.kwargs.append(kwargs)
        elif kwargs is None:
            self.kwargs.append({})
        else:
            raise ValueError("kwargs must be a dictionary or None")
        self.args.append(args)

    def run_process_queue(self, bio_data: Bio_Data) -> Bio_Data:

        bio_data = bio_data.copy()
        for i in range(len(self.process_list)):
            result = self.run_next(bio_data)
            bio_data = bio_data.join(result)
            self.processed_index += 1
        return bio_data

    def run_next(self, bio_data):
        bio_data = bio_data.copy()
        inputs = self.input_signals[self.processed_index]
        outputs = self.output_signals[self.processed_index]
        args = self.args[self.processed_index]
        kwargs = self.kwargs[self.processed_index]
        sampling_rate = kwargs.get("sampling_rate", None)
        output_sampling_rate = kwargs.get("new_sr", None)
        get_rate = False
        input_args = ()
        input_kwargs = {}
        n_windows = []
        if isinstance(inputs, dict):
            if sampling_rate is None:
                sampling_rate = []
                get_rate = True
            for key in inputs.keys():
                input_kwargs.append({key: bio_data[inputs[key]].channel})
                n_windows.append(bio_data[inputs[key]].n_windows)
                if get_rate:
                    sampling_rate.append(bio_data[inputs[key]].sampling_rate)
        elif isinstance(inputs, list):
            if sampling_rate is None:
                sampling_rate = []
                get_rate = True
            for i in inputs:
                input_args = input_args + (bio_data[i].channel,)
                n_windows.append(bio_data[i].n_windows)
                if get_rate:
                    sampling_rate.append(bio_data[i].sampling_rate)
        elif isinstance(inputs, str):
            if sampling_rate is None:
                sampling_rate = bio_data[inputs].sampling_rate
            input_args = (bio_data[inputs].channel,)
            n_windows.append(bio_data[inputs].n_windows)

        if any(n_windows[0] != n for n in n_windows):
            raise ValueError("All input channels must have the same number of windows")

        results = []
        if n_windows[0] == 1:
            current_args = input_args + args
            current_kwargs = {**input_kwargs, **kwargs}
            results = self.run_single(current_args, current_kwargs)
        else:
            for i in range(n_windows[0]):
                current_args = [x[i] for x in input_args]
                current_args = tuple(current_args)
                current_kwargs = {key: input_kwargs[key][i] for key in input_kwargs.keys()}
                current_args = current_args + args
                current_kwargs = {**current_kwargs, **kwargs}
                results.append(self.run_single(current_args, current_kwargs))

        output = self._handle_results(results, sampling_rate, outputs, n_windows[0], output_sampling_rate)
        return output

    def _handle_results(self, results, sampling_rate, name, n_windows, new_sr):

        if not new_sr is None:
            sampling_rate = new_sr
        if not self.is_event[self.processed_index]:
            results = np.array(results).reshape(len(name), n_windows, -1)
            results = np.squeeze(results)
            output = convert_channel(results, sampling_rate=sampling_rate, name=name, n_windows=n_windows)
        elif self.is_event[self.processed_index]:
            output = convert_event(results, sampling_rate=sampling_rate, name=name, n_windows=n_windows)

        return output

    def run_single(self, args, kwargs):

        result = self.process_list[self.processed_index].run(*args, **kwargs)

        return result

    def get_process_by_name(self, name):
        for process in self.process_list:
            if process.name == name:
                return process
        raise ValueError("Process with name " + name + " not found")

    def _process_io(self, input_signals, output_signals):

        """Check output signals and convert to list of list of strings"""
        if isinstance(output_signals, str):
            output_signals = output_signals
        elif isinstance(output_signals, list):
            if not all(isinstance(o, str) for o in output_signals):
                raise ValueError("output_signals must be a string or a list of strings")
            output_signals = output_signals
        elif isinstance(output_signals, dict):
            if not all(isinstance(o, str) for o in output_signals.values()):
                raise ValueError("output_signals must be a string or a list of strings")
            output_signals = output_signals
        else:
            raise ValueError("output_signals must be a string or a list of strings, or a dictionary with string values")

        self.output_signals.append(output_signals)

        # Check input signals
        if not isinstance(input_signals, (str, list, dict)):
            raise ValueError("input_signals must be a string, list of strings, or dictionary")

        if isinstance(input_signals, list):
            if not all(isinstance(o, str) for o in input_signals):
                raise ValueError("input_signals must be a string or a list of strings")
        elif isinstance(input_signals, dict):
            if not all(isinstance(o, str) for o in input_signals.values()):
                raise ValueError("input_signals must be a string or a list of strings")

        self.input_signals.append(input_signals)

    def __str__(self) -> str:
        representation = "Process list:\n"
        for i, p in enumerate(self.process_list):
            process = p.process_name
            if isinstance(self.input_signals[i], dict):
                process_in = self.input_signals[i].values()
            else:
                process_in = self.input_signals[i]
            if not isinstance(process_in, str):
                process_in = ",".join(process_in)
            if isinstance(self.output_signals[i], str):
                process_out = self.output_signals[i]
            elif isinstance(self.output_signals[i], list):
                process_out = ",".join(self.output_signals[i])
            representation += "\t" + str(i + 1) + ": " + process + "(" + process_in + ") -> " + process_out + "\n"

        return representation
