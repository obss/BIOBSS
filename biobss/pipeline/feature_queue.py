from __future__ import annotations

from copy import copy
from distutils.log import warn

import numpy as np
import pandas as pd

from .bio_data import Bio_Data
from .feature_extraction import Feature


class Feature_Queue:
    def __init__(self, name="Feature_Queue"):
        self.extraction_list = []
        self.input_signals = []
        self.args = []
        self.kwargs = []
        self.name = name
        self.windowed = False
        self.processed_index = 0
        self.feature_set = pd.DataFrame()
        self.prefix = []

    def add_feature(self, feature, input_signals=None, feature_prefix=None, *args, **kwargs):
        self.extraction_list.append(feature)
        self.input_signals.append(input_signals)
        self.args.append(args)
        self.kwargs.append(kwargs)
        self.prefix.append(feature_prefix)

    def run_feature_queue(self, bio_data: Bio_Data, reset=False) -> pd.DataFrame:
        bio_data = bio_data.copy()
        if reset:
            self.reset()
        for i in range(len(self.extraction_list)):
            res = self.run_next(bio_data)
            if self.prefix[i] is not None:
                res.columns = [self.prefix[i] + "_" + c for c in res.columns]
            self.feature_set = pd.concat([self.feature_set, res], axis=1)
            self.processed_index += 1

        return self.feature_set

    def run_next(self, bio_data: Bio_Data):
        inputs = self.input_signals[self.processed_index]
        args = self.args[self.processed_index]
        kwargs = self.kwargs[self.processed_index]
        if self.windowed:
            result = self.run_windowed(inputs, args, kwargs, bio_data)
        else:
            result = self.run_single(inputs, args, kwargs, bio_data)

        return result

    def run_single(self, inputs, args, kwargs, bio_data, index=None):
        input_keys = self._get_input_keys(inputs)

        if isinstance(inputs, dict):
            for key, value in inputs.items():
                if isinstance(value, str):
                    kwargs[key] = bio_data[value].channel[index] if index is not None else bio_data[value].channel
                elif isinstance(value, list):
                    kwargs[key] = [
                        bio_data[v].channel[index] if index is not None else bio_data[v].channel for v in value
                    ]
        elif isinstance(inputs, list):
            inputs = inputs[::-1]
            for i in inputs:
                if isinstance(i, list):
                    combined = self._combine_inputs(bio_data=bio_data, inputs=i, index=index)
                    args = (combined,) + args
                else:
                    args = (bio_data[i].channel[index] if index is not None else bio_data[i].channel,) + args
        elif isinstance(inputs, str):
            args = (bio_data[inputs].channel[index] if index is not None else bio_data[inputs].channel,) + args
        else:
            raise ValueError("Inputs must be a string, list, or dictionary.")

        result = self.extraction_list[self.processed_index].process(*args, **kwargs)
        result = self._process_results_single(result)
        result.index = [index]
        return result

    def run_windowed(self, inputs, args, kwargs, bio_data):
        n_windows = []
        results = []  # list of results for each window, to be concatenated

        input_keys = self._get_input_keys(inputs)
        for key in input_keys:
            if isinstance(key, str):
                n_windows.append(bio_data[key].n_windows)
            elif isinstance(key, list):
                input_windows = []
                for k in key:
                    input_windows.append(bio_data[k].n_windows)
                if not np.any(input_windows != input_windows[0]):
                    raise ValueError("All input signals must have the same number of windows.")
                else:
                    n_windows.append(input_windows[0])

        n_windows = min(n_windows)
        if n_windows > 1:
            for i in range(n_windows):
                results.append(self.run_single(inputs, args, kwargs, bio_data, i))
        else:
            results.append(self.run_single(inputs, args, kwargs, bio_data))
        results = pd.concat(results)
        return results

    def _get_input_keys(self, inputs):
        if isinstance(inputs, dict):
            input_keys = list(inputs.values())
        elif isinstance(inputs, str):
            input_keys = [inputs]
        elif isinstance(inputs, list):
            input_keys = inputs
        else:
            raise ValueError("Inputs must be a string, list, or dictionary.")

        return input_keys

    def _process_results(self, results):
        if isinstance(results, pd.DataFrame):
            return results
        elif isinstance(results, pd.Series):
            return results.to_frame()
        elif isinstance(results, dict):
            if not any(isinstance(v, dict) for v in results.values()):
                return pd.DataFrame([results])
            else:
                results = []
                for key, value in results.items():
                    results.append(self._process_results(value))
                return pd.concat(results, axis=1)
        elif isinstance(results, np.ndarray):
            if not results.ndim == 1:
                keys = self.generate_keys(results.shape[1])
                return pd.DataFrame([results], columns=keys)
            else:
                keys = self.generate_keys(1)
                return pd.DataFrame(results, columns=keys)
        elif isinstance(results, list):
            if np.dim(results) == 1:
                keys = self.generate_keys(1)
                return pd.DataFrame([results], columns=keys)
            else:
                keys = self.generate_keys(len(results))
                return pd.DataFrame(results, columns=keys)

        else:
            raise ValueError("Results must be a pandas DataFrame, Series, numpy array, or list.")

        # Add tuple support?

    def reset(self):
        self.processed_index = 0
        self.feature_set = pd.DataFrame()

    def copy(self):
        return copy(self)

    def set_name(self, value):
        self.name = value

    def generate_keys(self, n):
        p_name = self.extraction_list[self.processed_index].name
        keys = []
        for i in range(n):
            keys.append(f"{p_name}_{i}")
        return keys

    def _process_results_single(self, results, key=None):

        if isinstance(results, pd.DataFrame):
            return results
        elif isinstance(results, pd.Series):
            if results.name is None:
                if key is None:
                    key = self.generate_keys(1)
                else:
                    results.name = key
            return pd.DataFrame(results)
        elif isinstance(results, np.ndarray):
            if key is None:
                n_columns = results.shape[1]
                key = self.generate_keys(n_columns)
                if n_columns == 1:
                    key = [key]
            elif isinstance(key, str):
                key = [key]
            return pd.DataFrame(results, columns=[key])

        elif isinstance(results, (int, float)):
            if key is None:
                key = self.generate_keys(1)
            return pd.DataFrame([results], columns=[key])

        elif isinstance(results, (list, tuple)):
            if key is None:
                key = self.generate_keys(len(results))
                if len(results) == 1:
                    key = [key]
            return pd.DataFrame(results, columns=key)
        elif results is None:
            warn("Feature returned None")
            return pd.DataFrame()
        elif isinstance(results, dict):
            if any(isinstance(v, (list, tuple, np.ndarray, dict)) for v in results.values()):
                results_deeper = {}
                for k, v in results.items():
                    results_deeper[k] = self._process_results_single(v, k)
                results = pd.concat(results_deeper, axis=0)
                results_val = results.values
                results_cols = results.columns
                temp = pd.DataFrame([np.ones(len(results_cols))], columns=results_cols, dtype=object)
                if np.shape(results_val)[1] == len(results_cols) and np.shape(results_val)[0] != len(results_cols):
                    results_val = results_val.transpose()
                for i in range(len(results_cols)):
                    temp.iloc[0, i] = results_val[i]
                return temp
            else:
                return pd.DataFrame([results])
        else:
            raise ValueError("Feature returned an invalid type")

    def _combine_inputs(self, bio_data, inputs, index=None):
        input_args = []
        for i in inputs:
            if not isinstance(i, str):
                raise ValueError("Inputs must be a list of strings.")
            if index is not None:
                input_args.append(bio_data[i].channel[index])
            else:
                input_args.append(bio_data[i].channel)

        return np.array(input_args)
