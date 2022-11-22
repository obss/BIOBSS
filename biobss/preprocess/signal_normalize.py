from numpy.typing import ArrayLike
from ..pipeline.bio_channel import Bio_Channel

def normalize_signal(signal: ArrayLike, method: str='zscore') -> ArrayLike:
    """Normalizes a signal

    Args:
        signal (ArrayLike): Signal to be normalized
        method (str, optional): Normalization method. Defaults to 'zscore'.

    Raises:
        ValueError: If method is not 'zscore' or 'minmax'.

    Returns:
        ArrayLike: Normalized signal
    """
    # Need to add signal check
    if(method == 'zscore'):
        return (signal-signal.mean())/signal.std()
    elif(method == 'minmax'):
        return (signal-signal.min())/(signal.max()-signal.min())

    else:
        raise ValueError(
            f"Unknown method '{method}', available values are [zscore, minmax].")
