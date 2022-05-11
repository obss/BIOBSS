
from .signal import Signal

class Signal_Windows:
    def __init__(self):
        self.window_size = 0
        self.step_size = 0
        self.sampling_rate = 0
        self.signal_windows = []
        

    def __iter__(self):
        return self
    
    def create_from_segments(self,segments,window_size,step_size,sampling_rate):
        self.window_size = window_size
        self.step_size = step_size
        self.sampling_rate=sampling_rate
        self.signal_windows=segments
        return self
    
    def create_from_signal(self,signal,window_size,step_size,sampling_rate):
        self.window_size = window_size
        self.step_size = step_size
        self.sampling_rate=sampling_rate
        self.signal_windows=signal.segment_signal(self.window_size,self.step_size,sampling_rate=self.sampling_rate)
        return self
    
    def __next__(self):
        if self.window_size > 0:
            self.window_size -= self.step_size
            return self.window_size
        else:
            raise StopIteration
        
    def get_number_of_windows(self):
        return len(self.signal_windows)
    
    def __repr__(self) -> str:
        return "Signal_Windows(windows={},window_size={},step_size={},sampling_rate={})".format(self.signal_windows,self.window_size,self.step_size,self.sampling_rate)
    
 