import numpy as np
import scipy as sp
import pandas as pd
import neurokit2 as nk
from numpy.typing import ArrayLike
from ..signaltools import peakdetection
from ..edatools import hjorth

#These constants are defined considering the respiration range (6-60 breaths/min)
LAG_MIN = 1.33 #seconds. Period of respiration cycle for upper limit of respiration range.
LAG_MAX = 10 #seconds. Period of respiration cycle for lower limit of respiration range.

def extract_resp_sig(peaks_locs: ArrayLike, peaks_amp: ArrayLike, troughs_amp: ArrayLike, sampling_rate: float, mod_type: str=['AM','FM','BW'],resampling_rate: float=10) -> dict:
    """Extracts the respiratory signal using the modulations resulted from respiratory activity.

    Args:
        peaks_locs (ArrayLike): PPG signal peak locations
        peaks_amp (ArrayLike): PPG signal peak amplitudes
        troughs_amp (ArrayLike): PPG signal trough amplitudes
        sampling_rate (float): Sampling rate of the original PPG signal
        mod_type (Array, optional): Modulation type: 'AM' for amplitude modulation, 'FM' for frequency modulation, 'BW' for baseline wander. Defaults to ['AM','FM','BW'].
        resampling_rate (float, optional): Sampling rate of the extracted respiratory signal. Defaults to 10.

    Returns:
        dict: Dictionary of extracted respiratory signals.
    """

    info={}
    rs_ratio=int(sampling_rate/resampling_rate)

    if 'AM' in mod_type:
        #Amplitude modulation
        am=peaks_amp-troughs_amp[:-1]
        am_x=peaks_locs
        am_respsignal, am_xrs = _interpolate_and_resample(am, am_x, peaks_locs, rs_ratio, -1)

        info['am_y']=am_respsignal
        info['am_x']=am_xrs

    if 'FM' in mod_type:
        #Frequency modulation
        fm=np.diff(peaks_locs)/sampling_rate
        fm_x=peaks_locs[:-1]
        fm_respsignal, fm_xrs = _interpolate_and_resample(fm, fm_x, peaks_locs, rs_ratio, -2)

        info['fm_y']=fm_respsignal
        info['fm_x']=fm_xrs

    if 'BW' in mod_type:
        #Baseline wander
        bw=(peaks_amp+troughs_amp[:-1])/2
        bw_x=peaks_locs
        bw_respsignal, bw_xrs = _interpolate_and_resample(bw, bw_x, peaks_locs, rs_ratio, -1)
 
        info['bw_y']=bw_respsignal
        info['bw_x']=bw_xrs
      
    return info


def _interpolate_and_resample(sig, sig_x, peaks_locs, rs_ratio, offset):
    #Helper function for extracting respiratory signals.

    sig_interp=sp.interpolate.interp1d(sig_x,sig)
    num_x=peaks_locs[offset]-peaks_locs[0]+1
    sig_xnew=np.linspace(peaks_locs[0],peaks_locs[offset],num_x)
    sig_new=sig_interp(sig_xnew)
    respsig=sig_new[::rs_ratio]
    resp_x=sig_xnew[::rs_ratio].astype(int)
 
    return respsig, resp_x


def filter_resp_sig(resampling_rate: float=10, rsp_clean_method: str='khodadad2018', **kwargs) -> dict:
    """Filters extracted respiratory signal.

    Args:
        resampling_rate (float, optional): Resampling rate. Defaults to 10.

    kwargs:
        am_sig (Array, optional):Respiratory signal calculated from amplitude modulation. Defaults to None.
        am_x (Array, optional):x-axis values of amplitude modulation signal. Defaults to None.

        fm_sig (Array, optional):Respiratory signal calculated from frequency modulation. Defaults to None.
        fm_x (Array, optional):x-axis values of frequency modulation signal. Defaults to None.

        bw_sig (Array, optional):Respiratory signal calculated from baseline wander. Defaults to None.
        bw_x (Array, optional):x-axis values of baseline wander signal. Defaults to None.

    Returns:
        dict: Dictionary of filtered respiratory signals.
    """

    filtered_resp={}

    if 'am_sig' in kwargs.keys():
        cl_am=nk.rsp_clean(kwargs['am_sig'],sampling_rate=resampling_rate,method=rsp_clean_method)
        filtered_resp['am_sig']=cl_am
        filtered_resp['am_x']=kwargs['am_x']

    if 'fm_sig' in kwargs.keys():
        cl_fm=nk.rsp_clean(kwargs['fm_sig'],sampling_rate=resampling_rate,method=rsp_clean_method)
        filtered_resp['fm_sig']=cl_fm
        filtered_resp['fm_x']=kwargs['fm_x']

    if 'bw_sig' in kwargs.keys():
        cl_bw=nk.rsp_clean(kwargs['bw_sig'],sampling_rate=resampling_rate,method=rsp_clean_method)    
        filtered_resp['bw_sig']=cl_bw
        filtered_resp['bw_x']=kwargs['bw_x']
    
    return filtered_resp


def estimate_rr(resp_sig: ArrayLike, sampling_rate: float, method: str='peakdet',delta: float=0.001) -> float:
    """Estimates respiratory rate from the respiratory signal.

    Args:
        resp_sig (Array): Respiratory signal.
        sampling_rate (float): Sampling rate (resampled)
        method (str): Method for rr estimation: 'peakdet' or 'xcorr'. Defaults to 'peakdet'.
        delta (float, optional): Parameter of 'peakdet' method. Defaults to 0.001.

    Returns:
        float: Estimated respiratory rate
    """
    
    if  (method=='peakdet'):
        maxtab,mintab=peakdetection._peakdetection_peakdet(resp_sig, delta)

        if (maxtab.size==0) or (mintab.size==0):
            raise ValueError("No peaks or troughs found.Respiratory rate can not be estimated!")
        else:
            peaks_locs=maxtab[:,0].astype(int)
            intervals=abs(np.diff(peaks_locs))/sampling_rate #seconds
            mean_int=np.mean(intervals)
            mean_rr=60/mean_int #br/minutes
    
    elif  (method=='xcorr'):
        rr_inst = nk.rsp_rate(resp_sig, sampling_rate=sampling_rate, mod_type="xcorr")
        mean_rr=rr_inst.mean()
           
    else:
        raise ValueError("The method should be 'peakdet' or 'xcorr'")
   
    return mean_rr


def calc_rqi(resp_sig: ArrayLike, resampling_rate: float=10, rqi_method: ArrayLike=['autocorr','hjorth']) -> dict:
    """Calculates respiratory quality index for the given respiratory signal. 

    Args:
        resp_sig (Array): Respiratory signal
        resampling_rate (float, optional): Sampling rate (after resampling) of the respiratory signal. Defaults to 10.
        rqi_method (list, optional): Method for calculating respiratory quality index. Defaults to ['autocorr','hjorth'].

    Returns:
        dict: Dictionary of calculated RQIs.
    """

    rqindices={}

    if 'autocorr' in rqi_method:
        corr_coeff=[]
        sig=pd.Series(resp_sig)
        lag_min=int(LAG_MIN*resampling_rate) #seconds to samples
        lag_max=int(LAG_MAX*resampling_rate) #seconds to samples 
        
        for lag in range(lag_min,lag_max+1):
            c=sig.autocorr(lag=lag)
            corr_coeff.append(c)

        rqindices['autocorr']=max(corr_coeff)

    if 'hjorth' in rqi_method:
        hjorth_par=hjorth.get_hjorth_features(resp_sig)
        rqindices['hjorth']=hjorth_par['signal_complexity']
        
    return rqindices


def fuse_rr(fusion_method: str='SmartFusion', rqi: ArrayLike=None, **kwargs) -> float:
    """Fuses respiratory rates calculated from different modulation types.

    Args:
        fusion_method (str, optional): Fusion method. 'SmartFusion' of 'QualityFusion'. Defaults to 'SmartFusion'.
        rqi (Array, optional): Respiratory quality indices. Defaults to None.
        rr_am (float, optional): Respiratory rate calculated from amplitude modulation. Defaults to None.
        rr_fm (float, optional): Respiratory rate calculated from frequency modulation. Defaults to None.
        rr_bw (float, optional): Respiratory rate calculated from baseline wander. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        float: Fused respiratory rate
    """

    if kwargs:
        rr_est=[]
        
        for mod_type in kwargs.keys():
            rr_est.append(kwargs[mod_type])

        if fusion_method=='SmartFusion':                     
            rr_std=np.std(rr_est)

            if (rr_std <= 4 ):
                rr_fused=np.mean(rr_est)               
            else:  
                rr_fused=['invalid signal']

        elif fusion_method=='QualityFusion':
            if rqi is not None:
                rr_fused= np.sum(np.asarray(rqi)*np.asarray(rr_est))/np.sum(rqi)
            else:
                raise ValueError("RQI values are required.'")

        else:
            raise ValueError("The method should be 'SmartFusion' or 'QualityFusion'.") 

    else:
        raise ValueError("At least one respiratory rate value is required.'")

    return rr_fused

    


