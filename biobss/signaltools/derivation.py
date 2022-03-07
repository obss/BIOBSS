import numpy as np


def ppg_derivation(sig, second=True):
    """
    Calculates derivatives of the ppg signal
    Args:
        sig (_type_): input signal
        second (bool, optional): Returns second derivative of the ppg signal if True. Defaults to True.

    Returns:
        _type_: _description_
    """

    info={}

    vpg=np.gradient(sig) #axis=1 may be needed

    apg=np.gradient(vpg) #axis=1 may be needed

    info["VPG"]=vpg
    
    if second:

        info["APG"]=apg


    return info