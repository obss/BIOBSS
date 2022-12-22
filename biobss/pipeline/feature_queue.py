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
        self.kwargs=[]
        self.args=[]
        self.name = name
        self.processed_index = 0
        self.windowed = False
        

    def add_feature(self, feature: Feature, input_signals=None,*args,**kwargs):
        self.extraction_list.append(feature)
        self.kwargs.append(kwargs)
        self.args.append(args)
        if(isinstance(input_signals,(str,list,dict))):
            if(isinstance(input_signals,list)):
                if(not all(isinstance(o, str) for o in input_signals)):
                    raise ValueError("If input signals is a list, all elements must be strings")                
            
            self.input_signals.append(input_signals)
        else:
            raise ValueError("Input signals must be a string, list or dictionary")
        
        
    def run_feature_queue(self, bio_data: Bio_Data,feature_set:pd.DataFrame) -> Bio_Data:

        bio_data = bio_data.copy()
        for i in range(len(self.extraction_list)):
            res= self.run_next(bio_data)
            if(feature_set.empty):
                feature_set=res
            else:
                feature_set=feature_set.join(res)
        return feature_set
    
    
    def run_next(self,bio_data:Bio_Data):      
        bio_data=bio_data.copy()
        inputs = self.input_signals[self.processed_index]
        args=self.args[self.processed_index]
        kwargs=self.kwargs[self.processed_index]
        
        if(not self.windowed):
            if(isinstance(inputs,dict)):
                for key in inputs.keys():
                    kwargs.update({key:bio_data[inputs[key].channel]})
            elif(isinstance(inputs,list)):
                inputs=inputs[::-1]
                for i in inputs:
                    args=(bio_data[i].channel,)+args
            elif(isinstance(inputs,str)):
                args=(bio_data[inputs].channel,)+args
            
        else:
            pass
            
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
