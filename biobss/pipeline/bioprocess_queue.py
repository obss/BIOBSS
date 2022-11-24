from __future__ import annotations
from .bio_data import Bio_Data
from .bio_channel import Bio_Channel
from typing import Union

""" Process list object with add and iterate process objects"""


class Process_List():

    def __init__(self, name="Process_Queue"):
        self.process_list = []
        self.input_signals=[]
        self.name = name

    def add_process(self, process,input_signals=None, output_signals=None):
        self.process_list.append(process)
        if(isinstance(input_signals, str)):
            input_signals = [input_signals]
        elif(isinstance(input_signals, list)):
            input_signals = input_signals
        elif(input_signals is None):
            input_signals = ["ALL"]
        else:
            raise ValueError("input_signals must be a string or a list of strings")
        self.input_signals.append(input_signals)

    def get_process_by_name(self, name):
        for process in self.process_list:
            if(process.name == name):
                return process
        raise ValueError('Process with name '+name+' not found')

    def run_process_queue(self, bio_data: Bio_Data) -> Bio_Data:

        bio_data = bio_data.copy()
        
        for index,process in enumerate(self.process_list):
            c_info = bio_data.get_channel_names()
            for c_name in c_info:
                to_be_processed=False
                if(self.input_signals[index]==["ALL"]):
                    to_be_processed=True
                elif(c_name in self.input_signals[index]):
                    to_be_processed=True
                else:
                    to_be_processed=False
                    
                if(to_be_processed):
                    process_result = process.process(bio_data[c_name])
                    if(isinstance(process_result, Bio_Data)):
                            bio_data.join(process_result)
                    elif(isinstance(process_result, Bio_Channel)):
                            bio_data.add_channel(
                                process_result, c_name, modify_existed=True)
                    else:
                        pass

        return bio_data

    def __str__(self) -> str:
        representation = "Process list:\n"
        for i, p in enumerate(self.process_list):
            representation += "\t"+str(i+1)+": "+str(p)+"\n"

        return representation
