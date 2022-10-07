import numpy as np
from scipy import stats
import collections
from numpy.typing import ArrayLike

def calculate_shannon_entropy(sig: ArrayLike, base: int=2) -> float:
    """Calculates shannon entropy of the signal.

    Args:
        sig (ArrayLike): Signal to be analyzed.
        base (int): The logarithmic base to use, defaults to 2.

    Returns:
        float: Shannon entropy of the signal.
    """

    bases = collections.Counter([tmp_base for tmp_base in sig])
    # define distribution
    dist = [x/sum(bases.values()) for x in bases.values()]
 
    # use scipy to calculate entropy
    entropy_value = stats.entropy(dist, base=base)
 
    return entropy_value