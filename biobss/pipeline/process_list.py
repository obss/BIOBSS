
from .signal import Signal
from .signal_windows import Signal_Windows
from typing import Union

""" Process list object with add and iterate process objects"""
class Process_List():
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
    
    
    def run_process_queue(self,signal:Union[Signal,Signal_Windows]) -> Union[Signal,Signal_Windows]:
        signal=signal.copy()
        for process in self.process_list:
            if(isinstance(signal,Signal)):
                signal=process.process(signal)
            elif(isinstance(signal,Signal_Windows)):
                segments=[]
                for w in signal.signal_windows:
                    segments.append(process.process(w))        
                signal=Signal_Windows().create_from_segments(segments,signal.window_size,signal.step_size,signal.sampling_rate)

        return signal