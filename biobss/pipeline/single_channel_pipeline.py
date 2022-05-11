from git import RemoteProgress
from jsonschema import RefResolutionError
from .. import signaltools
from .process_list import Process_List
from .signal_windows import Signal_Windows
from .signal import Signal
"""a biological signal processing object with preprocessing and postprocessing steps"""

class Bio_Pipeline:
    """a biological signal processing object with preprocessing and postprocessing steps"""
    
    def __init__(self,features,modality="Generic",sigtype="Generic",windowed_process=False,window_size=None,step_size=None):
        if(windowed_process):
            self.windowed=True
            if(window_size is None or step_size is None):
                raise ValueError("window_size and step_size must be specified")
            else:
                self.set_window_parameters(window_size,step_size)
        else:
            self.window_size="Not Windowed"
            self.step_size="Not Windowed"
            self.windowed=False
        
        self.modality=modality
        self.sigtype=sigtype              
        self.preprocess_queue=Process_List(modality=modality,sigtype=sigtype)
        self.process_queue=Process_List(modality=modality,sigtype=sigtype)
        self.postprocess_queue=Process_List(modality=modality,sigtype=sigtype)
        
        
        
    def set_input(self,signal,sampling_rate,modality,name):
        if(modality!=self.modality):
            raise ValueError("Input modality does not match pipeline modality")
        
        self.input = signal
        self.sampling_rate = sampling_rate
        self.signal_name=name
        self.signal=Signal(signal,sampling_rate,modality,name)
        
        
    def set_window_parameters(self,window_size=10,step_size=5):
        self.window_size=window_size
        self.step_size=step_size
        
    def convert_windows(self):
        windows=signaltools.segment_signal(self.input,self.window_size,self.step_size,sampling_rate=self.sampling_rate)
        windows=Signal_Windows(windows,self.window_size,self.step_size,self.sampling_rate) 
        self.signal_windows=windows
        self.segmented=True
        
    def convert_windows(self,signal):       
        windows=signaltools.segment_signal_object(signal,self.window_size,self.step_size,sampling_rate=self.sampling_rate)
        return windows
        
    def create_feature_list(self,feature_list):
        self.feature_list=[]
        
    def run_pipeline(self):
        
        self.preprocessed_signal=self.preprocess_queue.run_process_queue(self.input)
        if(self.windowed):
            self.preprocessed_signal=self.convert_windows(self.preprocessed_signal)

        self.processed_signal=self.process_queue.run_process_queue(self.preprocessed_signal)
        
        
    def __repr__(self) -> str:
        representation="Bio_Pipeline:\n"
        representation+="\tModality: "+self.modality+"\n"
        representation+="\tSignal Type: "+self.sigtype+"\n"
        representation+="\tPreprocessors: "+str(self.preprocess_queue)+"\n"
        representation+="\tProcessors: "+str(self.process_queue)+"\n"
        representation+="\tPostprocessors: "+str(self.postprocess_queue)+"\n"
        representation+="\tWindow Size(Seconds): "+str(self.window_size)+"\n"
        representation+="\tStep Size: "+str(self.step_size)+"\n"
        return representation
        

        
    
    
    

