
def normalize_signal(signal,method='zscore'):
    if(method=='zscore'):
        return (signal-signal.mean())/signal.std()
    elif(method=='minmax'):
        return (signal-signal.min())/(signal.max()-signal.min())
    else:
        print("Normalization method error!")
        return None