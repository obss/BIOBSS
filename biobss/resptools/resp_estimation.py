import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy as sp
from numpy.typing import ArrayLike

from biobss.common import signal_hjorth
from biobss.preprocess import signal_detectpeaks

# These constants are defined considering the respiration range (6-60 breaths/min)
LAG_MIN = 1.33  # Period of respiration cycle for upper limit of respiration range (seconds).
LAG_MAX = 10  # Period of respiration cycle for lower limit of respiration range (seconds).


def extract_resp_sig(
    sig: ArrayLike,
    peaks_locs: ArrayLike,
    troughs_locs: ArrayLike,
    sampling_rate: float,
    mod_type: list = ["AM", "FM", "BW"],
    resampling_rate: float = 10.0,
) -> dict:
    """Extracts the respiratory signal(s) using the modulations resulted from respiratory activity and returns a dictionary of the respiratory signal(s).

    Args:
        sig (ArrayLike): Input signal (PPG or ECG).
        peaks_locs (ArrayLike): Input signal peak locations.
        troughs_locs (ArrayLike): Input signal trough locations.
        sampling_rate (float): Sampling rate of the input signal.
        mod_type (list, optional): Modulation type: 'AM' for amplitude modulation, 'FM' for frequency modulation, 'BW' for baseline wander. Defaults to ['AM','FM','BW'].
        resampling_rate (float, optional): Sampling rate of the extracted respiratory signal. Defaults to 10 (Hz).

    Raises:
        ValueError: If sampling rate is not greater than 0.
        ValueError: If resampling rate is not greater than 0.
        ValueError: If lengths of peak and trough location arrays do not match!

    Returns:
        dict: Dictionary of extracted respiratory signals.
    """

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    if resampling_rate <= 0:
        raise ValueError("Resampling rate must be greater than 0.")

    peaks_amp = sig[peaks_locs]
    troughs_amp = sig[troughs_locs]

    if len(peaks_locs) != len(troughs_locs) - 1:
        raise ValueError("Lengths of input arrays do not match!")

    mod_type = [x.upper() for x in mod_type]

    info = {}
    rs_ratio = int(sampling_rate / resampling_rate)

    if "AM" in mod_type:
        # Extracts the respiratory signal using amplitude modulation
        am = peaks_amp - troughs_amp[:-1]
        am_x = peaks_locs
        am_respsignal, am_xrs = _interpolate_and_resample(am, am_x, peaks_locs, rs_ratio, -1)

        info["am_y"] = am_respsignal
        info["am_x"] = am_xrs

    if "FM" in mod_type:
        # Extracts the respiratory signal using frequency modulation
        fm = np.diff(peaks_locs) / sampling_rate
        fm_x = peaks_locs[:-1]
        fm_respsignal, fm_xrs = _interpolate_and_resample(fm, fm_x, peaks_locs, rs_ratio, -2)

        info["fm_y"] = fm_respsignal
        info["fm_x"] = fm_xrs

    if "BW" in mod_type:
        # Extracts the respiratory signal using baseline wander
        bw = (peaks_amp + troughs_amp[:-1]) / 2
        bw_x = peaks_locs
        bw_respsignal, bw_xrs = _interpolate_and_resample(bw, bw_x, peaks_locs, rs_ratio, -1)

        info["bw_y"] = bw_respsignal
        info["bw_x"] = bw_xrs

    return info


def _interpolate_and_resample(sig: ArrayLike, sig_x: ArrayLike, peaks_locs: ArrayLike, rs_ratio: float, offset: int):
    """Interpolates and resamples respiratory signals."""

    sig_interp = sp.interpolate.interp1d(sig_x, sig)
    num_x = peaks_locs[offset] - peaks_locs[0] + 1
    sig_xnew = np.linspace(peaks_locs[0], peaks_locs[offset], num_x)
    sig_new = sig_interp(sig_xnew)
    respsig = sig_new[::rs_ratio]
    resp_x = sig_xnew[::rs_ratio].astype(int)

    return respsig, resp_x


def filter_resp_sig(resampling_rate: float = 10.0, rsp_clean_method: str = "khodadad2018", **kwargs) -> dict:
    """Filters extracted respiratory signal(s) and returns a dictionary of cleaned signal(s). Uses 'rsp_clean' function from the Neurokit2 library.

    Args:
        resampling_rate (float, optional): Resampling rate. Defaults to 10.
        rsp_clean_method (str, optional): Method to clean the respiratory signal. Defaults to 'khodadad2018'.

    kwargs:
        am_sig (Array, optional):Respiratory signal calculated from amplitude modulation. Defaults to None.
        am_x (Array, optional):x-axis values of amplitude modulation signal. Defaults to None.
        fm_sig (Array, optional):Respiratory signal calculated from frequency modulation. Defaults to None.
        fm_x (Array, optional):x-axis values of frequency modulation signal. Defaults to None.
        bw_sig (Array, optional):Respiratory signal calculated from baseline wander. Defaults to None.
        bw_x (Array, optional):x-axis values of baseline wander signal. Defaults to None.

    Returns:
        dict: Dictionary of filtered respiratory signals and x-axis values for each signal.
    """
    if resampling_rate <= 0:
        raise ValueError("Resampling rate must be greater than 0.")

    rsp_clean_method = rsp_clean_method.lower()

    filtered_resp = {}

    if "am_sig" in kwargs.keys():
        cl_am = nk.rsp_clean(kwargs["am_sig"], sampling_rate=resampling_rate, method=rsp_clean_method)
        filtered_resp["am_sig"] = cl_am
        filtered_resp["am_x"] = kwargs["am_x"]

    if "fm_sig" in kwargs.keys():
        cl_fm = nk.rsp_clean(kwargs["fm_sig"], sampling_rate=resampling_rate, method=rsp_clean_method)
        filtered_resp["fm_sig"] = cl_fm
        filtered_resp["fm_x"] = kwargs["fm_x"]

    if "bw_sig" in kwargs.keys():
        cl_bw = nk.rsp_clean(kwargs["bw_sig"], sampling_rate=resampling_rate, method=rsp_clean_method)
        filtered_resp["bw_sig"] = cl_bw
        filtered_resp["bw_x"] = kwargs["bw_x"]

    return filtered_resp


def estimate_rr(
    resp_sig: ArrayLike, resampling_rate: float = 10.0, method: str = "peakdet", delta: float = 0.001
) -> float:
    """Estimates respiratory rate from the respiratory signal.

    Args:
        resp_sig (ArrayLike): Respiratory signal.
        sampling_rate (float): Sampling rate of the respiratory signal (resampling rate, Hz). Defaults to 10.
        method (str): Method used for respiratory rate estimation. Can be one of 'peakdet' or 'xcorr'. Defaults to 'peakdet'.
                      Uses 'rsp_rate' function from the Neurokit2 library if the method is 'xcorr'.
        delta (float, optional): Parameter of 'peakdet' method. Defaults to 0.001.

    Returns:
        float: Estimated respiratory rate (breaths/minutes).
    """
    if resampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    method = method.lower()

    if method == "peakdet":
        mean_rr = _estimate_rr_peakdet(resp_sig, resampling_rate, delta=delta)

    elif method == "xcorr":
        mean_rr = _estimate_rr_xcorr(resp_sig, resampling_rate=resampling_rate)

    else:
        raise ValueError("The method should be one of 'peakdet' or 'xcorr'.")

    return mean_rr


def _estimate_rr_peakdet(resp_sig: ArrayLike, resampling_rate: float = 10.0, delta: float = 0.001) -> float:
    """Estimates respiratory rate using 'peakdet' method."""

    maxtab, mintab = signal_detectpeaks._peakdetection_peakdet(resp_sig, delta)

    if (maxtab.size == 0) or (mintab.size == 0):
        raise ValueError("No peaks or troughs found.Respiratory rate can not be estimated!")
    else:
        peaks_locs = maxtab[:, 0].astype(int)
        intervals = abs(np.diff(peaks_locs)) / resampling_rate  # seconds
        mean_int = np.mean(intervals)
        mean_rr = 60 / mean_int  # br/minutes

    return mean_rr


def _estimate_rr_xcorr(resp_sig: ArrayLike, resampling_rate: float = 10.0) -> float:
    """Estimates raspiratory rate using 'xcorr' method."""

    rr_inst = nk.rsp_rate(resp_sig, sampling_rate=resampling_rate, method="xcorr")
    mean_rr = rr_inst.mean()

    return mean_rr


def calc_rqi(resp_sig: ArrayLike, resampling_rate: float = 10.0, rqi_method: list = ["autocorr", "hjorth"]) -> dict:
    """Calculates respiratory quality index for the respiratory signal.

    Args:
        resp_sig (ArrayLike): Respiratory signal.
        resampling_rate (float, optional): Sampling rate (after resampling, Hz) of the respiratory signal. Defaults to 10.
        rqi_method (list, optional): Method for calculating respiratory quality index. Defaults to ['autocorr','hjorth'].

    Returns:
        dict: Dictionary of calculated RQIs.
    """
    if resampling_rate <= 0:
        raise ValueError("Resampling rate must be greater than 0.")

    rqi_method = [x.lower() for x in rqi_method]

    rqindices = {}

    if "autocorr" in rqi_method:
        corr_coeff = _calc_rqi_autocorr(resp_sig, resampling_rate)
        rqindices["autocorr"] = max(corr_coeff)

    if "hjorth" in rqi_method:
        rqindices["hjorth"] = signal_hjorth.hjorth_complexity_mobility(resp_sig)[0]

    return rqindices


def _calc_rqi_autocorr(resp_sig: ArrayLike, resampling_rate: float = 10.0) -> dict:
    """Calculates respiratory quality index using autocorrelation method."""

    corr_coeff = []
    sig = pd.Series(resp_sig)
    lag_min = int(LAG_MIN * resampling_rate)  # seconds to samples
    lag_max = int(LAG_MAX * resampling_rate)  # seconds to samples

    for lag in range(lag_min, lag_max + 1):
        c = sig.autocorr(lag=lag)
        corr_coeff.append(c)

    return corr_coeff


def fuse_rr(rr_est: ArrayLike, rqi: ArrayLike = None, fusion_method: str = "smartfusion") -> float:
    """Fuses respiratory rates calculated from different modulation types.

    Args:
        rr_est (Array): Respiratory rate estimations.
        rqi (Array, optional): Respiratory quality indices. Defaults to None.
        fusion_method (str, optional): Fusion method. Can be one of 'SmartFusion' of 'QualityFusion'. Defaults to 'SmartFusion'.

    Raises:
        ValueError: If lengths of 'rr_est' and 'rqi' arrays do not match.
        ValueError: If method is 'QualityFusion' and RQI values are not provided.
        ValueError: If method is not one of 'SmartFusion' and 'QualityFusion'.

    Returns:
        float: Fused respiratory rate
    """
    fusion_method = fusion_method.lower()

    if fusion_method == "smartfusion":
        rr_fused = _fuse_rr_smartfusion(rr_est)

    elif fusion_method == "qualityfusion":
        rr_fused = _fuse_rr_qualityfusion(rr_est, rqi)

    else:
        raise ValueError("The method should be one of 'SmartFusion' and 'QualityFusion'.")

    return rr_fused


def _fuse_rr_smartfusion(rr_est: ArrayLike) -> float:
    """Fuses respiratory rate estimations using 'SmartFusion' method."""
    rr_std = np.std(rr_est)

    if rr_std <= 4:
        rr_fused = np.mean(rr_est)
    else:
        rr_fused = ["invalid signal"]

    return rr_fused


def _fuse_rr_qualityfusion(rr_est: ArrayLike, rqi: ArrayLike = None) -> float:
    """Fuses respiratory rate estimations using 'QualityFusion' method."""

    if rqi is not None:
        if len(rr_est) != len(rqi):
            raise ValueError("The lengths of 'rr_est' and 'rqi' arrays do not match!")

        rr_fused = np.sum(np.asarray(rqi) * np.asarray(rr_est)) / np.sum(rqi)
    else:
        raise ValueError("RQI values are required for the 'QualityFusion' method.'")

    return rr_fused
