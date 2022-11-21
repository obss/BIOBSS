import numpy as np
import math
from numpy.typing import ArrayLike
from typing import Tuple

from biobss.sqatools.signal_quality import *

def ppg_sqa(ppg_sig: ArrayLike, sampling_rate:float, methods: list, **kwargs) -> bool:

    results = {}

    for method in methods:

        if method == 'flatline':
            info = detect_flatline_clipping(sig=ppg_sig, threshold=kwargs['flatline_threshold'], flatline=True, duration=kwargs['duration'])
            results['Flatline segments']=info['Flatline segments']

        elif method == 'clipping':
            info = detect_flatline_clipping(sig=ppg_sig, threshold=kwargs['clipping_threshold'], clipping=True)
            results['Clipped segments']=info['Clipped segments']

        elif method == 'physiological':
            info = check_phys(peaks_locs=kwargs['peaks_locs'], sampling_rate=sampling_rate)
            results['Physiological']=info

        elif method == 'morphological':
            info = check_morph(peaks_locs=kwargs['peaks_locs'],peaks_amps=kwargs['peaks_amp'],troughs_locs=kwargs['troughs_locs'],troughs_amps=kwargs['troughs_amp'],sampling_rate=sampling_rate)
            results['Morphological']=info

        elif method == 'template':
            info = template_matching(sig=ppg_sig,peaks_locs=kwargs['peaks_locs'])
            results['Template matching']=info

        else:
            raise ValueError("Undefined method for PPG signal quality assessment!")

    return results