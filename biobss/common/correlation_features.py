from scipy.stats import pearsonr
import itertools
from numpy.typing import ArrayLike


def correlation_features(signals: ArrayLike, signal_names: ArrayLike):
    """a function to calculate correlation of N signals by each other

    Args:
        signals (2-D Arraylike): Array of signals
        signal_names (Arraylike): Array of signal names

    Returns:
        dict : dict of correlation features
    """

    data = dict(zip(signal_names, signals))
    comb = list(itertools.combinations(signal_names, 2))
    corr_list = {}
    for i in comb:
        corr_list["".join(i) + "_correl"] = pearsonr(data[i[0]], data[i[1]])[0]

    return corr_list

