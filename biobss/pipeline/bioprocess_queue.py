from __future__ import annotations
from .bio_data import Bio_Data
from .bio_channel import Bio_Channel
from typing import Union
from .event_channel import Event_Channel
""" Process list object with add and iterate process objects"""
from warnings import warn


class Process_List():

    def __init__(self, name="Process_Queue"):
        self.process_list = []
        self.input_signals=[]
        self.output_signals=[]
        self.kwargs=[]
        self.args=[]
        self.name = name
        self.processed_index = 0

    def add_process(self, process,input_signals=None, output_signals=None,*args,**kwargs):
        self.process_list.append(process)
        self._process_io(input_signals,output_signals)
        if(isinstance(kwargs,dict)):
            self.kwargs.append(kwargs)
        elif(kwargs is None):
            self.kwargs.append({})
        else:
            raise ValueError("kwargs must be a dictionary or None")
        self.args.append(args)

    def run_process_queue(self, bio_data: Bio_Data) -> Bio_Data:

        bio_data = bio_data.copy()
        for i in range(len(self.process_list)):
            bio_data = self.run_next(bio_data)
        return bio_data
    
    def run_next(self,bio_data:Bio_Data):
        
        bio_data=bio_data.copy()
        inputs = self.input_signals[self.processed_index]
        args=self.args[self.processed_index]
        kwargs=self.kwargs[self.processed_index]
        
        if(isinstance(inputs,dict)):
            for key in inputs.keys():
                kwargs.update({key:bio_data[inputs[key]]})
        elif(isinstance(inputs,list)):
            for i in inputs:
                args=(bio_data[i],)+args
        elif(isinstance(inputs,str)):
            args=(bio_data[inputs],)+args
            
        result=self.process_list[self.processed_index].process(*args,**kwargs)    
        if(isinstance(result, Bio_Channel)):
            if(not isinstance(self.output_signals[self.processed_index],str)):
                if(len(self.output_signals[self.processed_index])!=1):
                    raise ValueError("Single channel output must be a string")
                else:
                    self.output_signals[self.processed_index]=self.output_signals[self.processed_index][0]
            result.signal_name=self.output_signals[self.processed_index]
            bio_data.add_channel(result)
        elif(isinstance(result, Event_Channel)):
            if(not isinstance(self.output_signals[self.processed_index],str)):
                if(len(self.output_signals[self.processed_index])!=1):
                    raise ValueError("Single channel output must be a string")
                else:
                    self.output_signals[self.processed_index]=self.output_signals[self.processed_index][0]
            result.signal_name=self.output_signals[self.processed_index]
            bio_data.add_event_channel(result)
        elif(isinstance(result, Bio_Data)):
            if(not isinstance(self.output_signals[self.processed_index],dict)):
                warn("If output of the process is Bio_Data, output signals argument must be a dictionary or it will bi ignored")
            else:
                if(len(self.output_signals[self.processed_index].values())!=result.channel_count):
                    raise ValueError("Output Bio_Data must have the same number of channels as the output_signals list")
            if(isinstance(self.output_signals[self.processed_index],dict)):
                for i in range(result.channel_count):
                    result[self.output_signals[self.processed_index].values()[i].keys()].signal_name=self.output_signals[self.processed_index].values()[i]
            else:
                pass
            bio_data.join(result)
            
        self.processed_index+=1
        return bio_data
            
    def _process_io(self,input_signals,output_signals):
        
        """ Check output signals and convert to list of list of strings"""
        if(isinstance(output_signals, str)):
            output_signals = output_signals
        elif(isinstance(output_signals, list)):
            if(not all(isinstance(o, str) for o in output_signals)):
                raise ValueError("output_signals must be a string or a list of strings")
            output_signals = output_signals
        elif(isinstance(output_signals,dict)):
            if(not all(isinstance(o, str) for o in output_signals.values())):
                raise ValueError("output_signals must be a string or a list of strings")
            output_signals = output_signals
        else:
            raise ValueError("output_signals must be a string or a list of strings, or a dictionary with string values")
                
        self.output_signals.append(output_signals)
        
        # Check input signals
        if(not isinstance(input_signals, (str,list,dict))):
            raise ValueError("input_signals must be a string, list of strings, or dictionary")

        if(isinstance(input_signals, list)):
            if(not all(isinstance(o, str) for o in input_signals)):
                raise ValueError("input_signals must be a string or a list of strings")
        elif(isinstance(input_signals,dict)):
            if(not all(isinstance(o, str) for o in input_signals.values())):
                raise ValueError("input_signals must be a string or a list of strings")
            
        self.input_signals.append(input_signals)
        

    def get_process_by_name(self, name):
        for process in self.process_list:
            if(process.name == name):
                return process
        raise ValueError('Process with name '+name+' not found')

    def __str__(self) -> str:
        representation = "Process list:\n"
        for i, p in enumerate(self.process_list):
            process= p.name
            if(isinstance(self.input_signals[i],dict)):
                process_in=self.input_signals[i].values()
            else:
                process_in=self.input_signals[i]                
            if(not isinstance(process_in,str)):
                process_in=",".join(process_in)   
            if(isinstance(self.output_signals[i],str)):              
                process_out=self.output_signals[i]
            elif(isinstance(self.output_signals[i],list)):
                process_out=",".join(self.output_signals[i])            
            representation += "\t"+str(i+1)+": "+process+"("+process_in+") -> "+process_out+"\n"

        return representation

