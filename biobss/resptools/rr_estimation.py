import numpy as np
import scipy as sp
import pandas as pd
import neurokit2 as nk
from numpy.typing import ArrayLike

from ..signaltools import peakdetection
from ..edatools import hjorth


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
    rs_len=int(sampling_rate/resampling_rate)

    if 'AM' in mod_type:
        #Amplitude modulation
        am=peaks_amp-troughs_amp[:-1]
        am_x=peaks_locs
        am_interp=sp.interpolate.interp1d(am_x,am)

        am_num=peaks_locs[-1]-peaks_locs[0]+1
        am_xnew=np.linspace(peaks_locs[0],peaks_locs[-1],am_num)
        am_new=am_interp(am_xnew)

        am_respsignal=am_new[::rs_len]
        am_xrs=am_xnew[::rs_len].astype(int)

        info['am_y']=am_respsignal
        info['am_x']=am_xrs

    if 'FM' in mod_type:

        #Frequency modulation
        fm=np.diff(peaks_locs)/sampling_rate
        fm_x=peaks_locs[:-1]
        fm_interp=sp.interpolate.interp1d(fm_x,fm)

        fm_num=peaks_locs[-2]-peaks_locs[0]+1
        fm_xnew=np.linspace(peaks_locs[0],peaks_locs[-2],fm_num)
        fm_new=fm_interp(fm_xnew)

        fm_respsignal=fm_new[::rs_len]
        fm_xrs=fm_xnew[::rs_len].astype(int)

        info['fm_y']=fm_respsignal
        info['fm_x']=fm_xrs

    if 'BW' in mod_type:

        #Baseline wander
        bw=(peaks_amp+troughs_amp[:-1])/2
        bw_x=peaks_locs
        bw_interp=sp.interpolate.interp1d(bw_x,bw)

        bw_num=peaks_locs[-1]-peaks_locs[0]+1
        bw_xnew=np.linspace(peaks_locs[0],peaks_locs[-1],bw_num)
        bw_new=bw_interp(bw_xnew)

        bw_respsignal=bw_new[::rs_len]
        bw_xrs=bw_xnew[::rs_len].astype(int)

        info['bw_y']=bw_respsignal
        info['bw_x']=bw_xrs
      

    return info




def filter_resp_sig(resampling_rate: float=10, **kwargs) -> dict:
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
        cl_am=nk.rsp_clean(kwargs['am_sig'],sampling_rate=resampling_rate,method='khodadad2018')
        filtered_resp['am_sig']=cl_am
        filtered_resp['am_x']=kwargs['am_x']

    if 'fm_sig' in kwargs.keys():
        cl_fm=nk.rsp_clean(kwargs['fm_sig'],sampling_rate=resampling_rate,method='khodadad2018')
        filtered_resp['fm_sig']=cl_fm
        filtered_resp['fm_x']=kwargs['fm_x']

    if 'bw_sig' in kwargs.keys():
        cl_bw=nk.rsp_clean(kwargs['bw_sig'],sampling_rate=resampling_rate,method='khodadad2018')    
        filtered_resp['bw_sig']=cl_bw
        filtered_resp['bw_x']=kwargs['bw_x']
    

    return filtered_resp



def estimate_rr(resp_sig: ArrayLike, sampling_rate: float, method: str='peakdet',delta: float=0.001) -> float:
    """Estimates respiratory rate from the respiratory signal.

    Args:
        resp_sig (Array): Extracted respiratory signal.
        sampling_rate (float): Sampling rate (resampled)
        method (str): method for rr estimation. 'peakdet', 'count_adv' or 'xcorr'. Defaults to 'peakdet'.
        delta (float, optional): Parameter of 'peakdet' method. Defaults to 0.001.

    Returns:
        float: Estimated respiratory rate
    """
    
    if  (method=='peakdet'):
        
        #peakdet
        maxtab,mintab=peakdetection._peakdetection_peakdet(resp_sig,delta)

        if (maxtab.size==0) or (mintab.size==0):
            mean_rr=[]
            print("No peaks or troughs found.Respiratory rate can not be estimated!")
            

        else:

            peaks_locs=maxtab[:,0].astype(int)
            peaks=maxtab[:,1]
            locs_troughs=mintab[:,0].astype(int)
            #troughs=mintab[:,1]

            intervals=abs(np.diff(peaks_locs))/sampling_rate #seconds
            mean_int=np.mean(intervals)
            mean_rr=60/mean_int #br/minutes

       
    elif  (method=='count_adv'):
    
        peaks_locs,_=sp.signal.find_peaks(resp_sig)
        peaks=resp_sig[peaks_locs]
    
        locs_troughs,_=sp.signal.find_peaks(-resp_sig)
        #troughs=resp_sig[locs_troughs]

        if (peaks_locs.size==0) or (locs_troughs.size==0):
            mean_rr=[]
            print("No peaks or troughs found.Respiratory rate can not be estimated!")
            
        
        else:
    
            dif=abs(np.diff(peaks))
            q3=np.percentile(dif,75)
            thresh=0.3*q3


            peak_ind=np.array([])

            stp=0
            while stp==0:
                ind=np.argmin(dif)
                if (dif[ind] < thresh):
                    peak_ind=np.append(peak_ind,ind)
            
                    dif[ind]=thresh+1
                    
                else:
                    stp=1  
            
            del_ind=np.concatenate((peak_ind,peak_ind+1))
            del_ind=np.unique(del_ind)
            

            peaks_locs=np.delete(peaks_locs,del_ind.astype(int))
            peaks=np.delete(peaks,del_ind.astype(int))

            intervals=abs(np.diff(peaks_locs))/sampling_rate #seconds
            mean_int=np.mean(intervals)
            mean_rr=60/mean_int #br/minutes
    
    elif  (method=='xcorr'):
        
        rr_inst = nk.rsp_rate(resp_sig, sampling_rate=sampling_rate, mod_type="xcorr")
        
        mean_rr=rr_inst.mean()
          
        
    else:
        raise ValueError("The method should be 'peakdet','count_adv' or 'xcorr'")
        
   
    return mean_rr



def calc_rqi(resp_sig: ArrayLike, resampling_rate: float=10, rqi_method: ArrayLike=['autocorr','hjorth']) -> dict:
    """Calculates respiratory quality index for the given respiratory signal. 

    Args:
        resp_sig (Array): Respiratory signal
        f_rs (float, optional): Sampling rate (after resampling) of the respiratory signal. Defaults to 10.
        rqi_method (list, optional): Method for calculating respiratory quality index. Defaults to ['autocorr','hjorth'].

    Returns:
        dict: Dictionary of calculated RQIs.
    """
    rqindices={}

    if 'autocorr' in rqi_method:

        corr_coeff=[]

        sig=pd.Series(resp_sig)
        lag_min=int(1.33*resampling_rate) #samples
        lag_max=int(10*resampling_rate) #samples 

        
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
        raise ValueError("At least one respiratory rate should be given.'")


    return rr_fused

    


