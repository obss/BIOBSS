from .. import signaltools
from process_list import Process_list
from signal_windows import Signal_Windows
from signal import Signal
"""a biological signal processing object with preprocessing and postprocessing steps"""

class BioPipeline:
    """a biological signal processing object with preprocessing and postprocessing steps"""
    
    def __init__(self,features,windowed_process=False,window_size=None,step_size=None):
        if(windowed_process):
            if(window_size is None or step_size is None):
                raise ValueError("window_size and step_size must be specified")
            else:
                self.set_window_parameters(window_size,step_size)
                       
        self.preprocess_queue=Process_list()
        self.process_queue=Process_list()
        self.postprocess_queue=Process_list()
        
        
        
    def set_input(self,signal,sampling_rate,modality,name):
        self.input = signal
        self.sampling_rate = sampling_rate
        self.modality=modality
        self.signal_name=name
        self.signal=Signal(signal,sampling_rate,modality,name)
        
        
    def set_window_parameters(self,window_size=10,step_size=5):
        self.window_size=window_size
        self.step_size=step_size
        
    def convert_windows(self):
        windows=signaltools.segment_signal(self.input,self.window_size,self.step_size,sampling_rate=self.sampling_rate)
        windows=Signal_Windows(windows,self.window_size,self.step_size,self.sampling_rate) 
        self.signal_windows=windows
        
    def convert_windows(self,signal):       
        windows=signaltools.segment_signal(signal,self.window_size,self.step_size,sampling_rate=self.sampling_rate)
        windows=Signal_Windows(windows,self.window_size,self.step_size,self.sampling_rate) 
        self.signal_windows=windows
        
    def run_pipeline(self):
        
        self.preprocessed_signal=self.preprocess_queue.run_process_queue(self.Signal)
        self.segmented_signal=self.convert_windows(self.preprocessed_signal)
        

        
    
    
    

