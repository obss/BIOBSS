from scipy import signal

def filter_signal(sig,filter_type,N,fs,f1=None,f2=None):
    """Filters the signal using a Butterworth filter

    Args:
        sig (array): Signal to be filtered
        filter_type (str): Low-pass, high-pass or band-pass 
        N (int): Filter order
        fs (float): Sampling rate
        f1 (float, optional): Lower cutoff frequency. Defaults to None.
        f2 (float, optional): Higher cutoff frequency. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        (array): filtered signal
    """

    if filter_type=='lowpass':

        W2=f2/(fs/2) #normalized frequency
        btype='lowpass'
        b2,a2 = signal.butter(N,W2,btype)
        filtered_sig=signal.filtfilt(b2,a2,sig,axis=0)    

    elif filter_type=='highpass':

        W1=f1/(fs/2) #normalized frequency
        btype='highpass'
        b1,a1 = signal.butter(N,W1,btype)
        filtered_sig=signal.filtfilt(b1,a1,sig,axis=0)

    elif filter_type=='bandpass':

        W2=f2/(fs/2) #normalized frequency
        btype='lowpass'
        b2,a2 = signal.butter(N,W2,btype)
        temp_sig=signal.filtfilt(b2,a2,sig,axis=0)       

        W1=f1/(fs/2) #normalized frequency
        btype='highpass'
        b1,a1 = signal.butter(N,W1,btype)
        filtered_sig=signal.filtfilt(b1,a1,temp_sig,axis=0)


    else:

        raise ValueError("Filter type error.")



    return filtered_sig


    