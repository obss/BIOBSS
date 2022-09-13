from .bio_data import Bio_Data
from .bio_channel import Bio_Channel
from typing import Union

""" Process list object with add and iterate process objects"""


class Process_List():

    def __init__(self, name="Process_Queue"):
        self.process_list = []
        self.name = name

    def add_process(self, process):
        self.process_list.append(process)

    def get_process_by_name(self, name):
        for process in self.process_list:
            if(process.name == name):
                return process
        raise ValueError('Process with name '+name+' not found')

    def run_process_queue(self, bio_data: Bio_Data) -> Bio_Data:

        bio_data = bio_data.copy()
        for process in self.process_list:
            c_info = bio_data.get_channel_names()
            for c_name in c_info:
                if process.check_input(bio_data[c_name]):
                    process_result = process.process(bio_data[c_name])
                    if(isinstance(process_result, Bio_Data)):
                        bio_data.join(process_result)
                    elif(isinstance(process_result, Bio_Channel)):
                        bio_data.add_channel(
                            process_result, c_name, modify_existed=True, modality=bio_data[c_name].signal_modality)
                else:
                    pass

        return bio_data

    def __str__(self) -> str:
        representation = "Process list:\n"
        for i, p in enumerate(self.process_list):
            representation += "\t"+str(i+1)+": "+str(p)+"\n"

        return representation
