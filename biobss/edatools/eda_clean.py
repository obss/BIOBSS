import neurokit2 as nk
import numpy as np


def eda_clean(eda_signal: np.ndarray, sampling_rate: float,method='neurokit'):


    if(method=='neurokit' or method=='biosppy'):        
        cleaned = nk.eda_clean(eda_signal, sampling_rate=sampling_rate, method=method)
    else:
        raise Exception("Method not implemented")

    return cleaned