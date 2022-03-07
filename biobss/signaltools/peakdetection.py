import numpy as np
from numpy import NaN, Inf, arange, isscalar, asarray, array
import pandas as pd
import heartpy as hp
import scipy as sp
import sys


def peakdetection(sig,fs, method='peakdet', delta=None):
    """Detects peaks and valleys of a signal using one of two methods.

    Args:
        sig (_type_): Signal to be analyzed
        fs (_type_): Sampling rate
        delta (_type_): Parameter of the peakdet method
        method (str, optional): 'peakdet' or 'heartpy'. Defaults to 'peakdet'.

    Raises:
        ValueError: _description_

    Returns:
        _type_: A dictionary containing peak locations, peak values, onset locations and onset values.
    """
    info={}

    if method=='peakdet':

        maxtab,mintab=_peakdetection_peakdet(sig,delta)

        locs_p=maxtab[:,0].astype(int)
        peaks=maxtab[:,1]
        locs_v=mintab[:,0].astype(int)
        valleys=mintab[:,1]

        info["Peak_locs"]= locs_p
        info["Peaks"]=peaks
        info["Onset_locs"]=locs_v
        info["Onsets"]=valleys


    elif method=='heartpy':

        wd_p=_peakdetection_heartpy(sig,fs)

        info["Peak_locs"] = wd_p['peaklist']
        info["Peaks"]=sig[wd_p['peaklist']]

        sig_v=max(sig)-sig

        wd_v=_peakdetection_heartpy(sig_v,fs)

        info["Onset_locs"] = wd_v['peaklist']
        info["Onsets"]=sig[wd_v['peaklist']]


    else:

        raise ValueError("Method should be 'peakdet' or 'heartpy'.")


    return info



def _peakdetection_peakdet(v, delta, x= None): 

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
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
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



    return array(maxtab), array(mintab)

# if __name__=="__main__":
#     from matplotlib.pyplot import plot, scatter, show
#     series = [0,0,0,2,0,0,0,-2,0,0,0,2,0,0,0,-2,0]
#     maxtab, mintab = peakdet(series,.3)
#     plot(series)
#     scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
#     scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
#     show()



def _peakdetection_heartpy(sig,fs):

    wd, m = hp.process(sig, sample_rate=fs)

    return wd




