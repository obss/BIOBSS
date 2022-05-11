from .signal import Signal

"""generic signal process object"""

class Bio_Process():
    
    def __init__(self,process_method,modality,sigtype,**kwargs) -> None:
        
        self.method=process_method
        self.modality=modality
        self.sigtype=sigtype
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
        signal=signal.copy()
        return self.method(signal,**self.kwargs)
    
    def get_name(self):
        return self.method.__name__

    
    

