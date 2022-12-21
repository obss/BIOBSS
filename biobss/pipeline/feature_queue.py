from __future__ import annotations
from distutils.log import warn
from .bio_data import Bio_Data
import pandas as pd
import numpy as np
from typing import Union
from .feature_extraction import Feature
from copy import copy


class Feature_Queue():
    
    def __init__(self,name="Feature_Queue"):
        self.extraction_list = []
        self.input_signals=[]
        self.output_signals=[]
        self.kwargs=[]
        self.args=[]
        self.name = name
        self.processed_index = 0
        

    def add_feature(self, feature: Feature, input_signals=None,*args,**kwargs):
        self.feature_queue[feature.name] = feature
        self.extraction_list.append(Feature)
        self._process_io(input_signals)
        if(isinstance(kwargs,dict)):
            self.kwargs.append(kwargs)
        elif(kwargs is None):
            self.kwargs.append({})
        else:
            raise ValueError("kwargs must be a dictionary or None")
        self.args.append(args)
        
    def run_process_queue(self, bio_data: Bio_Data,feature_set:pd.DataFrame) -> Bio_Data:

        bio_data = bio_data.copy()
        feature_data=copy(feature_set)
        for i in range(len(self.extraction_list)):
            feature_data.update = self.run_next(bio_data,feature_data)
        return feature_data
    
    def run_next(self,bio_data:Bio_Data,feature_set:pd.DataFrame):      
        bio_data=bio_data.copy()
        inputs = self.input_signals[self.processed_index]
        args=self.args[self.processed_index]
        kwargs=self.kwargs[self.processed_index]
        
        if(isinstance(inputs,dict)):
            for key in inputs.keys():
                kwargs.update({key:bio_data[inputs[key]]})
        elif(isinstance(inputs,list)):
            for i in inputs:
                args=([bio_data[i]],)+args
        elif(isinstance(inputs,str)):
            args=(bio_data[inputs],)+args
            
        result=self.extraction_list[self.processed_index].process(*args,**kwargs)       

        self.processed_index+=1
        return result
        
    def _process_io(self,input_signals):

            if(not isinstance(input_signals, (str,list,dict))):
                raise ValueError("input_signals must be a string, list of strings, or dictionary")
            if(isinstance(input_signals, list)):
                if(not all(isinstance(o, str) for o in input_signals)):
                    raise ValueError("input_signals must be a string or a list of strings")
            elif(isinstance(input_signals,dict)):
                if(not all(isinstance(o, str) for o in input_signals.values())):
                    raise ValueError("input_signals must be a string or a list of strings")
                
            self.input_signals.append(input_signals)
