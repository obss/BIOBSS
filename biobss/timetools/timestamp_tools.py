import numpy as np

def create_timestamp_signal(resolution,length,start,rate):
    if(start < 0):
        raise ValueError('Timestamp start must be greater than 0')
    
    if(resolution == 'ns'):
                timestamp_factor = 1/1e-9
    elif(resolution == 'ms'):
                timestamp_factor = 1/0.001
    elif(resolution == 's'):
                timestamp_factor = 1
    elif(resolution == 'min'):
                timestamp_factor = 60
    else:
        raise ValueError('resolution must be "ns","ms","s","min"')
        
    timestamp = (np.arange(length)/rate)*timestamp_factor
    timestamp = timestamp+start
    
    return timestamp
    
    
def check_timestamp(timestamp,timestamp_resolution,regularity_factor=1):
    
    if(timestamp_resolution == 'ns'):
            regularity_parameter = 1e-9
    elif(timestamp_resolution == 'ms'):
            regularity_parameter = 0.001
    elif(timestamp_resolution == 's'):
            regularity_parameter = 1
    elif(timestamp_resolution == 'min'):
            regularity_parameter = 60
    else:
        raise ValueError('timestamp_resolution must be "ns","ms","s","min"')
    
    regularity_parameter = regularity_parameter*regularity_factor
    
    if(np.any(np.diff(timestamp) < 0)):
        raise ValueError('Timestamp must be monotonic')
    if(np.diff(timestamp).std() > regularity_parameter):  # TODO: optimize this parameter
        raise ValueError('Timestamp must be regularly spaced')
    
    return True


