import sys
from typing import Any

import heartpy as hp
import numpy as np
from numpy.typing import ArrayLike
from scipy import signal


def peak_detection(sig: ArrayLike, sampling_rate: float, method: str = "peakdet", delta: float = None) -> dict:
    """Detects peaks and troughs of a signal returns a dictionary.

    Args:
        sig (ArrayLike): Signal to be analyzed.
        sampling_rate (float): Sampling rate of the signal (Hz).
        method (str, optional): Peak detection method. Should be one of 'peakdet', 'heartpy' and 'scipy'. Defaults to 'peakdet'.
                                See https://gist.github.com/endolith/250860 to get information about 'peakdet' method.
        delta (float, optional): Required parameter of the peakdet method.

    Raises:
        ValueError: If method is not one of 'peakdet', 'heartpy' and 'scipy'.

    Returns:
        dict: Dictionary of peak locations, peak amplitudes, trough locations and trough amplitudes.
    """
    method = method.lower()

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    info = {}

    if method == "peakdet":

        if delta is None:
            raise ValueError("Delta is required for 'peakdet' method.")

        maxtab, mintab = _peakdetection_peakdet(sig, delta)
        info["Peak_locs"] = maxtab[:, 0].astype(int)
        info["Trough_locs"] = mintab[:, 0].astype(int)

    elif method == "heartpy":

        wd_p = _peakdetection_heartpy(sig, sampling_rate)
        info["Peak_locs"] = wd_p["peaklist"]

        sig_t = max(sig) - sig
        wd_t = _peakdetection_heartpy(sig_t, sampling_rate)
        info["Trough_locs"] = wd_t["peaklist"]

    elif method == "scipy":

        peaks_locs = _peakdetection_scipy(sig)
        info["Peak_locs"] = peaks_locs

        sig_t = max(sig) - sig
        troughs_locs = _peakdetection_scipy(sig_t)
        info["Trough_locs"] = troughs_locs

    else:
        raise ValueError("Method should be one of 'peakdet' ,'heartpy' and 'scipy'.")

    return info


def _peakdetection_peakdet(v: ArrayLike, delta: float, x: ArrayLike = None) -> ArrayLike:
    """Detects signal peaks using the method 'peakdet'.
    Reference: https://gist.github.com/endolith/250860

    Converted from MATLAB script at http://billauer.co.il/peakdet.html

    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.

    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.

    """

    maxtab = []
    mintab = []

    if x is None:
        x = np.arange(len(v))

    v = np.asarray(v)

    if len(v) != len(x):
        sys.exit("Input vectors v and x must have same length")

    if not np.isscalar(delta):
        sys.exit("Input argument delta must be a scalar")

    if delta <= 0:
        sys.exit("Input argument delta must be positive")

    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN

    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)


def _peakdetection_heartpy(sig: ArrayLike, sampling_rate: float) -> Any:
    """Detects signal peaks using HeartPy."""
    wd, m = hp.process(sig, sample_rate=sampling_rate)

    return wd, m


def _peakdetection_scipy(sig: ArrayLike) -> Any:
    """Detects signal peaks using Scipy."""
    peaks_locs, _ = signal.find_peaks(sig)

    return peaks_locs
