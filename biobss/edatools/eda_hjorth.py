from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike

from biobss.common.signal_hjorth import *

def get_hjorth_features(signal: ArrayLike, prefix="signal"):
    """This method returns Hjörth parameters for the given signal.
    For more details, see the https://en.wikipedia.org/wiki/Hjorth_parameters

    Args:
        signal (ArrayLike): Input signal
        prefix (str, optional): prefix for signal name. Defaults to "signal".

    Returns:
        dict: calculated hjorth parameters
    """

    h_features = {}
    h_features[prefix + "_activity"] = calculate_activity(signal)
    h_features[prefix + "_complexity"], h_features[prefix + "_mobility"] = calculate_complexity(
        signal
    )
    return h_features