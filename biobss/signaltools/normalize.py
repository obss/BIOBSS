from numpy.typing import ArrayLike
from ..pipeline.bio_channel import Bio_Channel


def normalize_signal(signal: ArrayLike, method='zscore') -> ArrayLike:
    """Method for normalizing given signal

    Args:
        signal (arraylike): _description_
        method (str, optional): Normalization method. Defaults to 'zscore'.

    Returns:
        1-D array: normalized signal
    """

    # Need to add signal check
    if(method == 'zscore'):
        return (signal-signal.mean())/signal.std()
    elif(method == 'minmax'):
        return (signal-signal.min())/(signal.max()-signal.min())

    else:
        raise ValueError(
            f"Unknown method '{method}', available values are [zscore, minmax].")
