import math
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

from biobss.sqatools.signal_quality import *


def sqa_ppg(ppg_sig: ArrayLike, sampling_rate: float, methods: list, **kwargs) -> dict:
    """Assesses quality of PPG signal by applying rules based on morphological information.

    Args:
        ppg_sig (ArrayLike): Signal to be analyzed.
        sampling_rate (float): Sampling rate of the PPG signal (Hz).
        methods (list): Methods to be applied. It can be a list of 'flatline', 'clipping', 'physiological' and 'template'.
            'flatline': Detects beginning and end of flat segments.
            'clipping': Detects beginning and end of clipped segments.
            'physiological': Checks for physiological viability.
            'morphological': Checks for morphological limits.
            'template': Applies template matching method.

    Kwargs:
        threshold_pos (float): Threshold value for clipping detection.
        threshold_neg (float, optional):
        change_threshold (float): Threshold value for flatline detection.
        min_duration (float): Mimimum duration of flat segments for flatline detection.
        peaks_locs (float): R peak locations (sample).
        corr_th (float): Threshold for the correlation coefficient above which the signal is considered to be valid. Defaults to CORR_TH.

    Raises:
        ValueError: If method is undefined.
        ValueError: If 'change_threshold' and/or 'min_duration' is missing and the method 'flatline' is selected.
        ValueError: If 'threshold_pos' is missing and the method 'clipping' is selected.
        ValueError: If 'peaks_locs' is missing and the method 'physiological' is selected.
        ValueError: If 'peaks_locs' and/or 'troughs_locs' is missing and the method 'morphological' is selected.
        ValueError: If 'peaks_locs' is missing and the method 'template' is selected.

    Returns:
        dict: Dictionary of results for the applied methods.
    """

    results = {}

    for method in methods:

        if method == "flatline":
            if ("change_threshold" in kwargs) and ("min_duration" in kwargs):
                flatline_segments = detect_flatline_segments(
                    sig=ppg_sig, change_threshold=kwargs["change_threshold"], min_duration=kwargs["min_duration"]
                )
                results["Flatline segments"] = flatline_segments
            else:
                raise ValueError(
                    f"Missing keyword arguments 'change_threshold' and/or 'min_duration' for the selected method: {method}."
                )

        elif method == "clipping":
            if "threshold_pos" in kwargs:
                if "threshold_neg" in kwargs:
                    clipped_segments = detect_clipped_segments(
                        sig=ppg_sig, threshold_pos=kwargs["threshold_pos"], threshold_neg=kwargs["threshold_neg"]
                    )
                else:
                    clipped_segments = detect_clipped_segments(sig=ppg_sig, threshold_pos=kwargs["threshold_pos"])

                results["Clipped segments"] = clipped_segments
            else:
                raise ValueError(f"Missing keyword argument 'threshold_pos' for the selected method: {method}.")

        elif method == "physiological":
            if "peaks_locs" in kwargs:
                info = check_phys(peaks_locs=kwargs["peaks_locs"], sampling_rate=sampling_rate)
                results["Physiological"] = info
            else:
                raise ValueError(f"Missing keyword arguments 'peaks_locs' for the selected method: {method}.")

        elif method == "morphological":
            if ("peaks_locs" in kwargs) and ("troughs_locs" in kwargs):
                info = check_morph(
                    sig=ppg_sig,
                    peaks_locs=kwargs["peaks_locs"],
                    troughs_locs=kwargs["troughs_locs"],
                    sampling_rate=sampling_rate,
                )
                results["Morphological"] = info
            else:
                raise ValueError(
                    f"Missing keyword arguments 'peaks_locs' and/or 'troughs_locs' for the selected method: {method}."
                )

        elif method == "template":
            if "peaks_locs" in kwargs:
                if "corr_th" in kwargs:
                    info = template_matching(sig=ppg_sig, peaks_locs=kwargs["peaks_locs"], corr_th=kwargs["corr_th"])
                else:
                    info = template_matching(sig=ppg_sig, peaks_locs=kwargs["peaks_locs"])

                results["Template matching"] = info
            else:
                raise ValueError(f"Missing keyword arguments 'peaks_locs' for the selected method: {method}.")

        else:
            raise ValueError("Undefined method for PPG signal quality assessment!")

    return results
