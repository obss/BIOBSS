import neurokit2 as nk
from numpy.typing import ArrayLike


def filter_eda(sig: ArrayLike, sampling_rate: float, method: str = "neurokit") -> ArrayLike:
    """Filters EDA signal using predefined filter parameters.

    Args:
        sig (ArrayLike): EDA signal.
        sampling_rate (float): Sampling rate of the EDA signal (Hz).
        method (str, optional): Filtering method. It can be 'neurokit' or 'biosppy'. Defaults to 'neurokit'.

    Raises:
        Exception: If the method is not implemented.

    Returns:
        ArrayLike: Filtered EDA signal.
    """

    if method == "neurokit" or method == "biosppy":
        cleaned = nk.eda_clean(sig, sampling_rate=sampling_rate, method=method)
    else:
        raise Exception("Method not implemented.")

    return cleaned
