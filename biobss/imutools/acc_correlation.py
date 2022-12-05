from scipy.stats import pearsonr
import itertools
from numpy.typing import ArrayLike


def acc_corr_features(signals: ArrayLike, signal_names: ArrayLike, prefix: str="signal", **kwargs) -> dict:
    """Calculates correlation features for N signals

        For example:
        accx_accy_corr: correlation coefficient for x and y axes
        accx_accz_corr: correlation coefficient for x and z axes
        accy_accz_corr: correlation coefficient for y and z axes

    Args:
        signals (dict): Dictionary of signals for different axes.

    Returns:
        dict: Dictionary of correlation features
    """

    data = dict(zip(signal_names, signals))
    comb = list(itertools.combinations(signal_names, 2))
    
    corr_list = {}
    for i in comb:
        corr_list["_".join(i) + "_corr"] = pearsonr(data[i[0]], data[i[1]])[0]

    return corr_list
