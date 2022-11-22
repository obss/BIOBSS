from scipy.stats import pearsonr
import itertools


def get_corr_features(signals: dict, prefix: str="signal") -> dict:
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

    comb = list(itertools.combinations(signals.keys(), 2))
    corr_list = {}
    for i in comb:
        corr_list["_".join(i) + "_corr"] = pearsonr(signals[i[0]], signals[i[1]])[0]

    return corr_list
