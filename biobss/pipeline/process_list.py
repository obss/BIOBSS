from .bio_data import Bio_Data
from .data_channel import Data_Channel
from typing import Union

""" Process list object with add and iterate process objects"""
class Process_List():
    """ Process list object with add and iterate process objects"""
    def __init__(self,name="Process_Queue",modality="Generic",sigtype="Generic"):
        self.process_list=[]
        self.modality=modality
        self.sigtype=sigtype
        self.name=name
        
    def add_process(self,process):
        if(not process.check_modality(self.modality)):
            raise ValueError('Modality of process does not match modality of process list')
        elif(not process.check_sigtype(self.sigtype)):
            raise ValueError('Signal type of process does not match signal type of process list')
        else:
            self.process_list.append(process)
            
    def get_process_by_name(self,name):
        for process in self.process_list:
            if(process.name==name):
                return process
        raise ValueError('Process with name '+name+' not found')
    
    
    def run_process_queue(self,signal:Bio_Data) -> Bio_Data:

        signal=signal.copy()
        for process in self.process_list:
            c_info=signal.get_channel_names()
            for c_name in c_info:
                process_result=process.process(signal[c_name])
                if(isinstance(process_result,Bio_Data)):
                    signal.join(process_result)
                elif(isinstance(process_result,Data_Channel)):
                    signal.add_channel(process_result,c_name,modify_existed=True,modality=signal[c_name].signal_modality)


        return signal
    
    
    def __str__(self) -> str:
        representation="Process list:\n"
        for i,p in enumerate(self.process_list):
            representation+="\t"+str(i+1)+": "+str(p)+"\n"
            
        return representation