


class Signal():
    """ Signal object with add and iterate process objects"""
    def __init__(self,signal,sampling_rate,modality="Generic",sigtype="Generic"):
        self.signal=signal
        self.sampling_rate=sampling_rate
        self.modality=modality
        self.sigtype=sigtype

        
