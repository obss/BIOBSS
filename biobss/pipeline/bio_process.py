from __future__ import annotations
from .bio_data import Bio_Data
from .bio_channel import Channel
import pandas as pd
import numpy as np
import warnings
import inspect
from .event_channel import Event_Channel
from .channel_input import *
from .event_input import *
from .channel_output import *

"""generic signal process object"""


class Bio_Process:

    def __init__(self,process_method,process_name,*args,**kwargs):
        self.process_method = process_method
        self.process_name = process_name
        self.args = args
        self.kwargs =kwargs
        
    def process_args(self,**kwargs):
        """Process the input arguments"""        
        signature = inspect.signature(self.process_method)
        excess_args = []
        for key in kwargs.keys():
            if key not in signature.parameters.keys():
                excess_args.append(key)
        for e in excess_args:
            kwargs.pop(e)
        return kwargs
        
    
    def run(self,*args, **kwargs):
        """Run the process method on the input arguments"""
        args = args + self.args
        kwargs = {**self.kwargs,**kwargs}
        kwargs = self.process_args(**kwargs)
        result = self.process_method(*args,**kwargs)
        return result
