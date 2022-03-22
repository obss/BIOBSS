import numpy as np
from scipy import stats

def get_stat_features(signal,prefix="signal"):
    
    s_features={}
    min = np.min(signal)
    max = np.max(signal)
    s_features[prefix+"_mean"] = signal.mean()
    s_features[prefix+"_std"] = np.std(signal)
    s_features[prefix+"_max"] = max
    s_features[prefix+"_min"] = min
    s_features[prefix+"_drange"] = (max-min)
    s_features[prefix+"kusc"] = stats.kurtosis(signal)
    s_features[prefix+"sksc"] = stats.skew(signal)
    s_features[prefix+"mosc"] = stats.moment(signal, 2)
    
    return s_features



