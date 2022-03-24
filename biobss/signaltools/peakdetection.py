from typing import Any
import numpy as np
import heartpy as hp
import sys
from numpy.typing import ArrayLike

def peak_detection(sig: ArrayLike, fs: float, method: str='peakdet', delta: float=None) -> dict:
    """Detects peaks and troughs of a signal using one of two methods.

    Args:
        sig (array): Signal to be analyzed
        fs (float): Sampling rate
        delta (float): Parameter of the peakdet method
        method (str, optional): 'peakdet' or 'heartpy'. Defaults to 'peakdet'.

    Raises:
        ValueError: _description_

    Returns:
        (dict): Dictionary of peak locations, peak amplitudes, trough locations and trough amplitudes.
    """
    info={}

    if method=='peakdet':

        maxtab,mintab=_peakdetection_peakdet(sig,delta)

        locs_p=maxtab[:,0].astype(int)
        peaks=maxtab[:,1]
        locs_t=mintab[:,0].astype(int)
        troughs=mintab[:,1]

        info["Peak_locs"]= locs_p
        info["Peaks"]=peaks
        info["Trough_locs"]=locs_t
        info["Troughs"]=troughs


    elif method=='heartpy':

        wd_p=_peakdetection_heartpy(sig,fs)

        info["Peak_locs"] = wd_p['peaklist']
        info["Peaks"]=sig[wd_p['peaklist']]

        sig_t=max(sig)-sig

        wd_t=_peakdetection_heartpy(sig_t,fs)

        info["Trough_locs"] = wd_t['peaklist']
        info["Troughs"]=sig[wd_t['peaklist']]


    else:

        raise ValueError("Method should be 'peakdet' or 'heartpy'.")


    return info



def _peakdetection_peakdet(v: ArrayLike, delta: float, x: ArrayLike= None) -> ArrayLike: 

    """
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
        sys.exit('Input vectors v and x must have same length')
    
    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
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
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True



    return np.array(maxtab), np.array(mintab)



def _peakdetection_heartpy(sig: ArrayLike, fs: float) -> Any:

    wd, m = hp.process(sig, sample_rate=fs)

    return wd, m




