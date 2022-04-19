
from signal import Signal
""" Process list object with add and iterate process objects"""
class Process_list():
    """ Process list object with add and iterate process objects"""
    def __init__(self,modality="Generic",sigtype="Generic"):
        self.process_list=[]
        self.modality=modality
        self.sigtype=sigtype
        
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
    
    
    def run_process_queue(self,signal:Signal):
        for process in self.process_list:
            signal=process.process(signal)
        return signal