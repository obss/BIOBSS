from scipy.stats import pearsonr
import itertools


def get_corr_features(signals: dict, prefix: str="signal") -> dict:
    """Calculates correlation of N signals with each other.

    Args:
        signals (2-D Arraylike): Array of signals
        signal_names (Arraylike): Array of signal names

    Returns:
        dict : dict of correlation features
    """

    comb = list(itertools.combinations(signals.keys(), 2))
    corr_list = {}
    for i in comb:
        corr_list["_".join(i) + "_corr"] = pearsonr(signals[i[0]], signals[i[1]])[0]

    return corr_list
