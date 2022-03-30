import numpy as np
import scipy as sp
import neurokit2 as nk
from numpy.typing import ArrayLike

from ..signaltools import peakdetection



def extract_resp_sig(peaks_locs: ArrayLike, peaks_amp: ArrayLike, troughs_amp: ArrayLike, sampling_rate: float, mod_type: str='All',resampling_rate: float=10) -> dict:
    """Calculates the modulations resulted from respiratory activity.

    Args:
        peaks_locs (ArrayLike): PPG signal peak locations
        peaks_amp (ArrayLike): PPG signal peak amplitudes
        troughs_amp (ArrayLike): PPG signal trough amplitudes
        sampling_rate (float): Sampling rate of the original PPG signal
        mod_type (str, optional): Modulation type: 'AM' for amplitude modulation, 'FM' for frequency modulation, 'BW' for baseline wander. Defaults to 'All'.
        resampling_rate (float, optional): Sampling rate of the extracted respiratory signal. Defaults to 10.

    Returns:
        dict: Dictionary of extracted respiratory signals.
    """

     
    info={}
    rs_len=int(sampling_rate/resampling_rate)

    #Amplitude modulation
    am=peaks_amp-troughs_amp[:-1]
    am_x=peaks_locs
    am_interp=sp.interpolate.interp1d(am_x,am)
    am_num=peaks_locs[-1]-peaks_locs[0]+1
    am_xnew=np.linspace(peaks_locs[0],peaks_locs[-1],am_num)
    am_new=am_interp(am_xnew)

    am_respsignal=am_new[::rs_len]
    am_xrs=am_xnew[::rs_len]
    am_xrs=am_xrs.astype(int)

    #Frequency modulation
    fm=np.diff(peaks_locs)/sampling_rate
    fm_x=peaks_locs[:-1]
    fm_interp=sp.interpolate.interp1d(fm_x,fm)
    fm_num=peaks_locs[-2]-peaks_locs[0]+1
    fm_xnew=np.linspace(peaks_locs[0],peaks_locs[-2],fm_num)
    fm_new=fm_interp(fm_xnew)

    fm_respsignal=fm_new[::rs_len]
    fm_xrs=fm_xnew[::rs_len]
    fm_xrs=fm_xrs.astype(int)

    #Baseline wander
    bw=(peaks_amp+troughs_amp[:-1])/2
    bw_x=peaks_locs
    bw_interp=sp.interpolate.interp1d(bw_x,bw)
    bw_num=peaks_locs[-1]-peaks_locs[0]+1
    bw_xnew=np.linspace(peaks_locs[0],peaks_locs[-1],bw_num)
    bw_new=bw_interp(bw_xnew)

    bw_respsignal=bw_new[::rs_len]
    bw_xrs=bw_xnew[::rs_len]
    bw_xrs=bw_xrs.astype(int)

    if mod_type=='AM':

        info['Resp. signal']=am_respsignal
        info['x']=am_xrs

    elif mod_type=='FM':

        info['Resp. signal']=fm_respsignal
        info['x']=fm_xrs

    elif mod_type=='BW':

        info['Resp. signal']=bw_respsignal
        info['x']=bw_xrs

    else:
        info['AM']=am_respsignal
        info['Am_x']=am_xrs
        info['FM']=fm_respsignal
        info['FM_x']=fm_xrs
        info['BW']=bw_respsignal
        info['BW_x']=bw_xrs        

    return info




def filter_resp(y_am: ArrayLike=None, y_fm: ArrayLike=None, y_bw: ArrayLike=None, resampling_rate: float=10) -> dict:
    """Filters extracted respiratory signal

    Args:
        y_am (Array, optional):Respiratory signal calculated from amplitude modulation. Defaults to None.
        y_fm (Array, optional):Respiratory signal calculated from frequency modulation. Defaults to None.
        y_bw (Array, optional):Respiratory signal calculated from baseline wander. Defaults to None.
        resampling_rate (float, optional): Resampling rate. Defaults to 10.

    Returns:
        Dict: Dictionary of filtered respiratory signals.
    """

    filtered_resp={}

    if y_am is not None:
        cl_am=nk.rsp_clean(y_am,sampling_rate=resampling_rate,mod_type='khodadad2018')
        filtered_resp['AM']=cl_am

    if y_fm is not None:
        cl_fm=nk.rsp_clean(y_fm,sampling_rate=resampling_rate,mod_type='khodadad2018')
        filtered_resp['FM']=cl_fm

    if y_bw is not None:
        cl_bw=nk.rsp_clean(y_bw,sampling_rate=resampling_rate,mod_type='khodadad2018')    
        filtered_resp['BW']=cl_bw
    

    return filtered_resp



def estimate_rr(resp_sig: ArrayLike, sampling_rate: float, mod_type: str='peakdet',delta: float=0.001) -> float:
    """Estimates respiratory rate from the respiratory signal.

    Args:
        resp_sig (Array): Extracted respiratory signal.
        sampling_rate (float): Sampling rate (resampled)
     mod_type (str): mod_type of rr estimation. 'peakdet', 'count_adv' or 'xcorr'. Defaults to 'peakdet'.
        delta (float, optional): Parameter of 'peakdet' mod_type. Defaults to 0.001.

    Returns:
        float: Estimated respiratory rate
    """
    
    if  (mod_type=='peakdet'):
        
        #peakdet
        maxtab,mintab=peakdetection._peakdetection_peakdet(resp_sig,delta)

        if (maxtab.size==0) or (mintab.size==0):
            print("Respiratory rate can not be estimated!")
            mean_rr=[]

        else:

            peaks_locs=maxtab[:,0].astype(int)
            peaks=maxtab[:,1]
            locs_troughs=mintab[:,0].astype(int)
            #troughs=mintab[:,1]

            intervals=abs(np.diff(peaks_locs))/sampling_rate #seconds
            mean_int=np.mean(intervals)
            mean_rr=60/mean_int #br/minutes

       
    elif  (mod_type=='count_adv'):
    
        peaks_locs,_=sp.signal.find_peaks(resp_sig)
        peaks=resp_sig[peaks_locs]
    
        locs_troughs,_=sp.signal.find_peaks(-resp_sig)
        #troughs=resp_sig[locs_troughs]

        if (peaks_locs.size==0) or (locs_troughs.size==0):
            print("Respiratory rate can not be estimated!")
            mean_rr=[]
        
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
    
    elif  (mod_type=='xcorr'):
        
        rr_inst = nk.rsp_rate(resp_sig, sampling_rate=sampling_rate, mod_type="xcorr")
        
        mean_rr=rr_inst.mean()
          
        
    else:
        print("The mod_type should be 'peakdet','count_adv' or 'xcorr'")
        

    
    return mean_rr


def fuse_rr(rr_am: float=None, rr_fm: float=None, rr_bw: float=None, fusion_method: str='Smart') -> float:
    """Fuses respiratory rates calculated from different modulation types.

    Args:
        rr_am (float, optional): Respiratory rate calculated from amplitude modulation. Defaults to None.
        rr_fm (float, optional): Respiratory rate calculated from frequency modulation. Defaults to None.
        rr_bw (float, optional): Respiratory rate calculated from baseline wander. Defaults to None.
        fusion_method (str, optional): Fusion mod_type. Defaults to 'Smart' (Smart Fusion).

    Returns:
        _type_: _description_
    """
    if rr_am is None:
        rr_am=[]
    if rr_fm is None:
        rr_fm=[]
    if rr_bw is None:
        rr_bw=[]


    if fusion_method=='Smart':
        #Smart Fusion
        rr_est=np.array([rr_am,rr_fm,rr_bw])
        rr_std=np.std(rr_est)

        if (rr_std <= 4 ):
            rr_fused=np.mean(rr_est)            
            
        else:
            
            rr_fused=[]

    return rr_fused

    


