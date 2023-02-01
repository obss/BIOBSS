import collections

from numpy.typing import ArrayLike
from scipy import stats


def calculate_shannon_entropy(sig: ArrayLike, base: int = 2) -> float:
    """Calculates shannon entropy of a signal.
    Entropy of a signal X(t) is defined as:
    S(X) = -sum(p(xi)*log2(p(xi)))
    xi: discrete values in X(t)
    p(xi): probability of obtaining xi

    Args:
        sig (ArrayLike): Signal to be analyzed.
        base (int): The logarithmic base to use, defaults to 2.

    Returns:
        float: Shannon entropy of the signal.
    """
    # count values in sig
    value_counts = collections.Counter([value for value in sig])
    # calculate probabilities
    pk = [x / sum(value_counts.values()) for x in value_counts.values()]
    # use scipy.stats to calculate entropy
    entropy_value = stats.entropy(pk=pk, base=base)

    return entropy_value
