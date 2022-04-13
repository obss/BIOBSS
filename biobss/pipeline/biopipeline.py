"""a biological signal processing object with preprocessing and postprocessing steps"""

class BioPipeline:
    """a biological signal processing object with preprocessing and postprocessing steps"""
    
    def __init__(self, preprocessors, postprocessors,features):
        self.preprocessors = preprocessors
        self.features = features
        self.postprocessors = postprocessors
        
    def set_input(self,signal,sampling_rate,modality):
        self.input = signal
        self.sampling_rate = sampling_rate
        self.modality=modality
        
        if(len(self.channels)==1):
            self.sigtype='singlechannel'
        else:
            self.sigtype='multichannel'

    def add_preprocessing_steps(self, preprocessors):
        self.preprocessors.extend(preprocessors)
        
        
    def list_preprocessing_steps(self):
        """return a list of the preprocessing steps"""
        return self.preprocessors

    def preprocessing(self):
        """process the signal with the preprocessors"""
        signal=self.input
        for preprocessor in self.preprocessors:
            signal = preprocessor.process(signal)        
        self.preprocessed_signal = signal
        
    
    def add_postprocessing_steps(self, postprocessors):
        self.postprocessors.extend(postprocessors)
        
    def postprocessing(self):
        """process the signal with the postprocessors"""
        signal=self.preprocessed_signal
        for postprocessor in self.postprocessors:
            signal = postprocessor.process(signal)
        self.postprocessed_signal = signal
    
    def extract_features(self):
        """extract features from the postprocessed signal"""
        features = {}
        for feature in self.features:
            features.update(feature.process(self.postprocessed_signal))
                
        return features
        
    def process(self, signal):
        """process the signal with the preprocessors and postprocessors"""
        self.set_input(signal,self.sampling_rate,self.channels,self.modality)
        self.preprocessing()
        self.postprocessing()
        return self.postprocessed_signal
    
    

