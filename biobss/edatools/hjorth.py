import numpy as np
from numpy.typing import ArrayLike

def get_hjorth_features(signal:ArrayLike, prefix="signal"):
    """This method returns Hjörth parameters for the given signal.
    For more details, see the https://en.wikipedia.org/wiki/Hjorth_parameters

    Args:
        signal (ArrayLike): Input signal
        prefix (str, optional): prefix for signal name. Defaults to "signal".

    Returns:
        dict: calculated hjorth parameters
    """

    h_features = {}
    h_features[prefix + "_activity"] = activity(signal)
    h_features[prefix + "_complexity"], h_features[prefix + "_mobility"] = complexity(
        signal
    )
    return h_features


def activity(signal):
    """ This method returns Hjörth activity for the given signal.

    Args:
        signal (arraylile): input signal

    Returns:
        _type_: _description_
    """
    
    return np.var(signal)



def mobility(signal):
    """ This method returns Hjörth mobility for the given signal.
    """
    f_derivative = np.gradient(signal, edge_order=1)
    mobility = np.square(np.var(f_derivative) / np.var(signal))
    return mobility


def complexity(signal):
    """ This method returns Hjörth complexity and mobility for the given signal.

    Mobility : The ratio of the variance of the first derivative of the signal to the variance of the signal.
    Complexity : The ratio of the variance of the second derivative of the signal to the variance of the signal.
    
    Args:
        signal (arraylike): input signal

    Returns:
        _type_: _description_
    """
    _mobility = mobility(signal)
    f_derivative = np.gradient(signal, edge_order=1)
    complexity = mobility(f_derivative) / _mobility
    return complexity, _mobility

