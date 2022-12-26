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
        self.feature_set=feature_set
        for i in range(len(self.extraction_list)):
            res= self.run_next(bio_data)
            res= self._process_results(res)
            if(feature_set.empty):
                feature_set=res
            else:
                feature_set=feature_set.join(res)
            self.processed_index+=1
            
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
        return result
        
    def _process_results(self,results,key=None):
        
        if(not self.windowed):
            return self._process_results_single(results)
        else:
            return self.process_results_windowed(results)     
               
    def _process_results_single(self,results,key=None):
        
        if(isinstance(results,pd.DataFrame)):
            return results
        elif(isinstance(results,pd.Series)):
            if(results.name is None):
                if(key is None):
                    key=self.generate_keys(1)
                else:
                    results.name=key
            return pd.DataFrame(results)
        elif(isinstance(results,np.ndarray)):
            if(key is None):
                n_columns = results.shape[1]
                key =self.generate_keys(n_columns)
                if(n_columns==1):
                    key=[key]
            else:
                return pd.DataFrame(results,columns=key)
        elif(isinstance(results,(int,float))):
            if(key is None):
                key = self.generate_keys(1)
            return pd.DataFrame([results],columns=[key])
        
        elif(isinstance(results,(list,tuple))):
            if(key is None):
                key = self.generate_keys(len(results))
                if(len(results)==1):
                    key=[key]
            return pd.DataFrame(results,columns=key)
        elif(results is None):
            warn("Feature returned None")
            return pd.DataFrame()
        elif(isinstance(results,dict)):
            if(any(isinstance(v, (list, tuple, np.ndarray,dict)) for v in results.values())):
                results_deeper = {}
                for k,v in results.items():
                    results_deeper[k]=self._process_results_single(v,k)
                results = pd.concat(results_deeper,axis=0)
                results_val = results.values
                results_cols = results.columns
                temp = pd.DataFrame([np.ones(len(results_cols))],columns = results_cols,dtype=object)
                for i in range(len(results_cols)):
                    temp.iloc[0,i]=results_val[i]
                return temp
            else:
                return pd.DataFrame([results])     
        else:
            raise ValueError("Feature returned an invalid type")
    

    
    
    def process_results_windowed(self,results):
        
                
        return 
    
    def generate_keys(self,n):
        
        feature_name = self.extraction_list[self.processed_index].name
        keys = [feature_name+"_"+str(i) for i in range(n)]
        return keys
        
