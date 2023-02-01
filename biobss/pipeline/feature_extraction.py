from __future__ import annotations

import inspect
from distutils.log import warn

import pandas as pd


class Feature:
    def __init__(self, name, function, *args, **kwargs):

        self.name = name
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def process(self, *args, **kwargs) -> pd.DataFrame:

        args = args + self.args
        kwargs.update(self.kwargs)
        kwargs = self._process_args(**kwargs)
        feature_output = self.__extract(*args, **kwargs)
        return feature_output

    def _process_args(self, **kwargs):

        signature = inspect.signature(self.function)
        excess_args = []
        for key in kwargs.keys():
            if key not in signature.parameters.keys():
                excess_args.append(key)
        for e in excess_args:
            kwargs.pop(e)
        return kwargs

    def __extract(self, *args, **kwargs):

        result = self.function(*args, **kwargs)
        return result

    def __str__(self) -> str:
        return self.name
