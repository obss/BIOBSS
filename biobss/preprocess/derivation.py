import numpy as np
from numpy.typing import ArrayLike


def ppg_derivation(sig: ArrayLike, second: bool=True, axis: int=None, edge_order: int=1) -> dict:
    """Computes first and second derivatives of a PPG signal.

    Args:
        sig (ArrayLike): A vector of values corresponding to PPG signal 
        second (bool, optional): Returns second derivative of the ppg signal if True. Defaults to True.
        axis (_type_, optional): The axis alongh which gradient is calculated. Defaults to None.
        edge_order (int, optional): Gradient is calculated using N-th order accurate differences at the boundaries. Defaults to 1.

    Returns:
        dict: A dictionary containing the first and second derivatives in the form of vector of values.
    """

    info = {}
    vpg = np.gradient(sig, axis=axis, edge_order=edge_order)
    info["VPG"] = vpg

    if second:
        apg = np.gradient(vpg, axis=axis, edge_order=edge_order)
        info["APG"] = apg

    return info
