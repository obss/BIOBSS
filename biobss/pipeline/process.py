from signal import Signal

"""generic signal process object"""

class Bio_process():
    
    def __init__(self,method,modality,**kwargs) -> None:
        self.method=method
        self.modality=modality
        self.kwargs=kwargs
        
        
    def check_modality(self,modality):
        if(self.modality==modality):
            return True
        else:
            return False
        
    def check_sigtype(self,sigtype):
        if(self.sigtype==sigtype):
            return True
        else:
            return False
        
    def process(self,signal:Signal):
        signal.signal=self.method(signal.signal,**self.kwargs)
    
    def get_name(self):
        return self.method

    
    

