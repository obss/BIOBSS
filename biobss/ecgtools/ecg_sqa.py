from numpy.typing import ArrayLike

from biobss.sqatools.signal_quality import *

def ecg_sqa(ecg_sig: ArrayLike, sampling_rate:float, methods: list, **kwargs) -> dict:
    """Assesses quality of ECG signal by applying rules based on morphological information.

    Args:
        ecg_sig (ArrayLike): Signal to be analyzed.
        sampling_rate (float): Sampling rate of the ECG signal (Hz).
        methods (list): Methods to be applied. It can be a list of 'flatline', 'clipping', 'physiological' and 'template'.
            'flatline': Detects beginning and end of flat segments.
            'clipping': Detects beginning and end of clipped segments.
            'physiological': Checks for physiological viability.
            'template': Applies template matching method. 

    Kwargs:
        duration (float): Mimimum duration of flat segments for flatline detection.
        clipping_threshold (float): Threshold value for clipping detection.
        flatline_threshold (float): Threshold value for flatline detection.
        peaks_locs (float): R peak locations (sample).
        corr_th (float): Threshold for the correlation coefficient above which the signal is considered to be valid. Defaults to CORR_TH.
    
    Raises:
        ValueError: If method is undefined.

    Returns:
        dict: Dictionary of results for the applied methods. 
    """
    results = {}

    for method in methods:

        if method == 'flatline':
            if ('flatline_threshold' in kwargs) and ('duration' in kwargs):
                info = detect_flatline_clipping(sig=ecg_sig, threshold=kwargs['flatline_threshold'], flatline=True, duration=kwargs['duration'])
                results['Flatline segments']=info['Flatline segments']
            else:
                raise ValueError(f"Missing keyword arguments 'flatline_threshold' and/or 'duration' for the selected method: {method}.")

        elif method == 'clipping':
            if 'clipping_threshold' in kwargs:
                info = detect_flatline_clipping(sig=ecg_sig, threshold=kwargs['clipping_threshold'], clipping=True)
                results['Clipped segments']=info['Clipped segments']
            else:
                raise ValueError(f"Missing keyword argument 'clipping_threshold' for the selected method: {method}.")

        elif method == 'physiological':
            if 'peaks_locs' in kwargs:
                info = check_phys(peaks_locs=kwargs['peaks_locs'], sampling_rate=sampling_rate)
                results['Physiological']=info
            else:
                raise ValueError(f"Missing keyword arguments 'peaks_locs' for the selected method: {method}.")

        elif method == 'template':
            if 'peaks_locs' in kwargs:
                if 'corr_th' in kwargs:
                    info = template_matching(sig=ecg_sig, peaks_locs=kwargs['peaks_locs'], corr_th=kwargs['corr_th'])
                else:
                    info = template_matching(sig=ecg_sig, peaks_locs=kwargs['peaks_locs'])
                    
                results['Template matching']=info
            else:
                raise ValueError(f"Missing keyword arguments 'peaks_locs' for the selected method: {method}.")

        else:
            raise ValueError(f"Undefined method {method} for ECG signal quality assessment!")

    return results