import numpy as np
from numpy.typing import ArrayLike

def ppg_derivation(sig: ArrayLike, second: bool=True, axis=None, edge_order=1) -> dict:
    """
    Calculates derivatives of the ppg signal
    Args:
        sig (array): ppg signal
        second (bool, optional): Returns second derivative of the ppg signal if True. Defaults to True.

    Returns:
        info (dict): dictionary of first and second derivatives
    """

    info={}

    vpg=np.gradient(sig, axis=axis, edge_order=edge_order ) 

    

    info["VPG"]=vpg
    
    if second:
        apg=np.gradient(vpg, axis=axis, edge_order=edge_order) 
        info["APG"]=apg

    return info