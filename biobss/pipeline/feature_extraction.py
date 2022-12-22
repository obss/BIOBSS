from __future__ import annotations
from distutils.log import warn
from .bio_data import Bio_Data
import pandas as pd
import numpy as np
from typing import Union
import inspect


class Feature():

    def __init__(self, name, function, *args, **kwargs):

        self.name = name
        self.function = function
        self.args = args
        self.kwargs=kwargs
    
    def process(self,*args,**kwargs) -> pd.DataFrame:

        args=args+self.args
        kwargs.update(self.kwargs)
        kwargs=self._process_args(**kwargs)
        feature_output = self.__extract(*args,**kwargs)
        return feature_output


    def _process_args(self,**kwargs):
             
        signature = inspect.signature(self.function)
        excess_args = []
        for key in kwargs.keys():
            if key not in signature.parameters.keys():
                excess_args.append(key)
        for e in excess_args:
            kwargs.pop(e)
        return kwargs


    def _process_results(self,results):
        if(isinstance(results,pd.DataFrame)):
            return results
        
        elif(isinstance(results,np.ndarray)):
            return pd.DataFrame(results)        
        elif(isinstance(results,dict)):
            return pd.DataFrame(results,orient='index')
        
        return results
    def __extract(self, *args,**kwargs) -> pd.DataFrame:
        
        result = self.function(*args,**kwargs)
        return result    
    
    
    

    def __str__(self) -> str:
        return self.name