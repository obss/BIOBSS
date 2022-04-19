

class Signal_Windows:
    def __init__(self,windows,window_size, step_size,sampling_rate):
        self.window_size = window_size
        self.step_size = step_size
        self.sampling_rate=sampling_rate
        self.signal_windows = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.window_size > 0:
            self.window_size -= self.step_size
            return self.window_size
        else:
            raise StopIteration
        
    def get_number_of_windows(self):
        return len(self.signal_windows)