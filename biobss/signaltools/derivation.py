import numpy as np


def ppg_derivation(sig, second=True):
    """
    Calculates derivatives of the ppg signal
    Args:
        sig (array): ppg signal
        second (bool, optional): Returns second derivative of the ppg signal if True. Defaults to True.

    Returns:
        info (dict): dictionary of first and second derivatives
    """

    info={}

    vpg=np.gradient(sig) #axis=1 may be needed

    

    info["VPG"]=vpg
    
    if second:
        apg=np.gradient(vpg) #axis=1 may be needed
        info["APG"]=apg

    return info