from __future__ import annotations
from distutils.log import warn
from .bio_data import Bio_Data
import pandas as pd
import numpy as np
from typing import Union


class Feature():

    def __init__(self, name, function, *args, **kwargs):

        self.name = name
        self.function = function
        self.args = args
        self.kwargs=kwargs
    
    def run(self,*args,**kwargs) -> pd.DataFrame:

        args=args+self.args
        kwargs.update(self.kwargs)
        kwargs=self._process_args(**kwargs)
        feature_output = self.__extract(*args,**kwargs)
        feature_output = self._process_results(feature_output)
        return feature_output


    def _process_results(self,results):
        if(isinstance(results,pd.DataFrame)):
            return results
        elif(isinstance(results,np.ndarray)):
            return pd.DataFrame(results)
        
        return results
    def __extract(self, *args,**kwargs) -> pd.DataFrame:
        pass


    def __str__(self) -> str:
        return self.name