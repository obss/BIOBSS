import numpy as np
import math
from numpy.typing import ArrayLike
from typing import Tuple

from biobss.sqatools.signal_quality import *

def sqa_ppg(ppg_sig: ArrayLike, sampling_rate:float, methods: list, **kwargs) -> dict:
    """Assesses quality of PPG signal by applying rules based on morphological information.

    Args:
        ppg_sig (ArrayLike): Signal to be analyzed.
        sampling_rate (float): Sampling rate of the PPG signal (Hz).
        methods (list): Methods to be applied.

    Raises:
        ValueError: If method is undefined.

    Returns:
        dict: Dictionary of boolean results of the applied rules.
    """

    results = {}

    for method in methods:

        if method == 'flatline':
            if ('flatline_threshold' in kwargs) and ('duration' in kwargs):
                info = detect_flatline_clipping(sig=ppg_sig, threshold=kwargs['flatline_threshold'], flatline=True, duration=kwargs['duration'])
                results['Flatline segments']=info['Flatline segments']
            else:
                raise ValueError(f"Missing keyword arguments 'flatline_threshold' and/or 'duration' for the selected method: {method}.")

        elif method == 'clipping':
            if 'clipping_threshold' in kwargs:
                info = detect_flatline_clipping(sig=ppg_sig, threshold=kwargs['clipping_threshold'], clipping=True)
                results['Clipped segments']=info['Clipped segments']
            else:
                raise ValueError(f"Missing keyword argument 'clipping_threshold' for the selected method: {method}.")

        elif method == 'physiological':
            if 'peaks_locs' in kwargs:
                info = check_phys(peaks_locs=kwargs['peaks_locs'], sampling_rate=sampling_rate)
                results['Physiological']=info
            else:
                raise ValueError(f"Missing keyword arguments 'peaks_locs' for the selected method: {method}.")

        elif method == 'morphological':
            if ('peaks_locs' in kwargs) and ('troughs_locs' in kwargs):
                info = check_morph(sig=ppg_sig, peaks_locs=kwargs['peaks_locs'], troughs_locs=kwargs['troughs_locs'], sampling_rate=sampling_rate)
                results['Morphological']=info
            else:
                raise ValueError(f"Missing keyword arguments 'peaks_locs' and/or 'troughs_locs' for the selected method: {method}.")

        elif method == 'template':
            if 'peaks_locs' in kwargs:
                if 'corr_th' in kwargs:
                    info = template_matching(sig=ppg_sig,peaks_locs=kwargs['peaks_locs'], corr_th=kwargs['corr_th'])
                else:
                    info = template_matching(sig=ppg_sig,peaks_locs=kwargs['peaks_locs'])

                results['Template matching']=info
            else:
                raise ValueError(f"Missing keyword arguments 'peaks_locs' for the selected method: {method}.")

        else:
            raise ValueError("Undefined method for PPG signal quality assessment!")

    return results