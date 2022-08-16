from typing import Dict, List
import numpy as np
from numpy.typing import ArrayLike
import warnings

from ..signaltools import filter_signal


DATA_TO_METRIC = {'PIM':['FXYZ_modified','UFM_modified','UFNM','FMpost_modified','FMpre'], 
                  'ZCM':['FXYZ','UFM','UFNM','FMpost','FMpre'], 
                  'TAT':['FXYZ','UFM','UFNM','FMpost','FMpre'], 
                  'MAD':['UFXYZ','FXYZ','UFM','UFNM','FMpost','FMpre'], 
                  'ENMO':['UFM'], 
                  'HFEN':['SpecialXYZ','SpecialM'], 
                  'AI': ['UFXYZ','FXYZ']}
    
METRIC_FUNCTIONS = {'PIM': lambda sig, dim, sampling_rate, _0, _1, triaxial: _calc_pim(sig, dim, sampling_rate, triaxial),
                      'ZCM': lambda sig, dim, _0, threshold, _1, triaxial: _calc_zcm(sig, dim, threshold, triaxial),
                      'TAT': lambda sig, dim, sampling_rate, threshold, _0, triaxial: _calc_tat(sig, dim, sampling_rate, threshold, triaxial),
                      'MAD': lambda sig, dim, _0, _1, _2, triaxial: _calc_mad(sig, dim, triaxial),
                      'ENMO': lambda sig, _0, _1, _2, _3, _4: _calc_enmo(sig),
                      'HFEN': lambda sig, dim, _0, _1, _2, triaxial: _calc_hfen(sig, dim, triaxial),
                      'AI': lambda sig, _0, _1, _2, baseline_variance, _4: _calc_ai(sig, baseline_variance)
                     }

DATASET_FUNCTIONS = {'UFXYZ': lambda sig_x,sig_y,sig_z,sampling_rate: generate_dataset(sig_x,sig_y,sig_z,sampling_rate,False,None,False,False,False),
                     'UFM': lambda sig_x,sig_y,sig_z,sampling_rate: generate_dataset(sig_x,sig_y,sig_z,sampling_rate,False,None,True,False,False),
                     'UFM_modified': lambda sig_x,sig_y,sig_z,sampling_rate: generate_dataset(sig_x,sig_y,sig_z,sampling_rate,False,None,True,False,True),
                     'UFNM': lambda sig_x,sig_y,sig_z,sampling_rate: generate_dataset(sig_x,sig_y,sig_z,sampling_rate,False,None,True,True,False),
                     'FXYZ': lambda sig_x,sig_y,sig_z,sampling_rate: generate_dataset(sig_x,sig_y,sig_z,sampling_rate,True,'pre',False,False,False),
                     'FXYZ_modified': lambda sig_x,sig_y,sig_z,sampling_rate: generate_dataset(sig_x,sig_y,sig_z,sampling_rate,True,'pre',False,False,True),
                     'FMpre': lambda sig_x,sig_y,sig_z,sampling_rate: generate_dataset(sig_x,sig_y,sig_z,sampling_rate,True,'pre',True,False,False),
                     'SpecialXYZ': lambda sig_x,sig_y,sig_z,sampling_rate: generate_dataset(sig_x,sig_y,sig_z,sampling_rate,True,'pre',False,False,False,'highpass',4,0.2),
                     'SpecialM': lambda sig_x,sig_y,sig_z,sampling_rate: generate_dataset(sig_x,sig_y,sig_z,sampling_rate,True,'pre',True,False,False,'highpass',4,0.2),
                     'FMpost': lambda sig_x,sig_y,sig_z,sampling_rate: generate_dataset(sig_x,sig_y,sig_z,sampling_rate,True,'post',True,False,False),
                     'FMpost_modified': lambda sig_x,sig_y,sig_z,sampling_rate: generate_dataset(sig_x,sig_y,sig_z,sampling_rate,True,'post',True,False,True)
                    }


def calc_activity_index(accx: ArrayLike, accy: ArrayLike, accz: ArrayLike, signal_length: float, sampling_rate: float, metric: str, input_types: list=None, threshold:List=None, baseline_variance:List=None, triaxial:bool=False) -> Dict:
    """Calculates the given activity index for the desired input types.

    Args:
        accx (ArrayLike): Acceleration vector for the x-axis.
        accy (ArrayLike): Acceleration vector for the y-axis.
        accz (ArrayLike): Acceleration vector for the z-axis.
        signal_length (float): Signal length in seconds.
        sampling_rate (float): Sampling rate of the acceleration signal(s).
        input_types (list): Type of dataset. Depends on the preprocessing methods applied on the raw acceleration data.
            UFXYZ : [accx, accy, accz]
            FXYZ : filter_signal([accx, accy, accz])
            FMpre : magnitude(filter_signal([accx, accy, accz]))    
            UFM : magnitude([accx, accy, accz])
            UFNM : normalize(magnitude([accx, accy, accz]))
            FMpost : filter_signal(magnitude([accx, accy, accz]))
            Special : filter_special([accx, accy, accz])
            FXYZ_modified = absolute(FXYZ)
            FMpost_modified = absolute(FMpost)
            UFM_modified = absolute (UFM - integral(gravity))
        metric (str): The activity index to be calculated.
        threshold (List, optional): Threshold level in g. This parameter is required for the 'ZCM' and 'TAT' metrics. Defaults to None.
        baseline_variance (List, optional): Baseline variance, corresponding to the variance of acceleration signal at rest (no movement).
                                             This parameter is required for the 'AI' metric. Defaults to None.
        triaxial (bool, optional): Parameter to decide if triaxial metrics should be combined into a single metric or not. Defaults to False.

    Raises:
        ValueError: If the input type is not one of valid types for the desired metric.

    Returns:
        Dict: A dictionary of calculated metric for the desired input types.
    """
    if input_types is None:
        input_types = DATA_TO_METRIC[metric]

    valid_inputs = DATA_TO_METRIC[metric]
    metric_function = METRIC_FUNCTIONS[metric]
    act_ind = {}
  
    for input_type in input_types:
        
        if input_type not in valid_inputs:
            raise ValueError("Invalid input type for the metric to be calculated!")
        else:  
            dataset_function = DATASET_FUNCTIONS[input_type]
            sig = dataset_function(accx, accy, accz, sampling_rate)
            dim=int(np.size(sig)/(signal_length*sampling_rate))

            if metric in ['ZCM', 'TAT'] and threshold is None:
                warnings.warn("Threshold level is required for this activity index, but not provided. Standard deviation of the signal will be used as threshold.")
                threshold = calc_threshold(sig, dim, input_type)

            act = metric_function(sig, dim, sampling_rate, threshold, baseline_variance, triaxial)
            if not triaxial:
                act_ind[input_type] = act[0]
            else:
                act_axes={}
                act_axes['x'] = act[0]
                act_axes['y'] = act[1]
                act_axes['z'] = act[2]
                act_ind[input_type] = act_axes
    
    return act_ind
            
def generate_dataset(accx: ArrayLike, accy:ArrayLike, accz:ArrayLike, sampling_rate:float, filtering:bool=False, filtering_order:str=None, magnitude:bool=False, normalize:bool=False, modify:bool=False, filter_type:str='bandpass', N:int=2, f1:float=0.5, f2:float=2) -> ArrayLike:
    """Generates datasets by applying appropriate preprocessing steps to the raw acceleration signals.

        The datasets are:
            UFXYZ : [accx, accy, accz]
            FXYZ : filter_signal([accx, accy, accz])
            FMpre : magnitude(filter_signal([accx, accy, accz]))    
            UFM : magnitude([accx, accy, accz])
            UFNM : normalize(magnitude([accx, accy, accz]))
            FMpost : filter_signal(magnitude([accx, accy, accz]))
            Special : filter_special([accx, accy, accz])
            FXYZ_modified = absolute(FXYZ)
            FMpost_modified = absolute(FMpost)
            UFM_modified = absolute (UFM - integral(gravity))

    Args:
        accx (ArrayLike): Acceleration vector for the x-axis.
        accy (ArrayLike): Acceleration vector for the y-axis.
        accz (ArrayLike): Acceleration vector for the z-axis.
        sampling_rate (float): Sampling rate of the acceleration signal(s).
        filtering (bool, optional): Parameter to decide if filtering should be applied or not. Defaults to False.
        filtering_order (str, optional): The order of filtering, should be 'pre', 'post' or 'None'. Defaults to None.
        magnitude (bool, optional): Parameter to decide if magnitude of the signals should be calculated or not. Defaults to False.
        normalize (bool, optional): Parameter to decide if the signal should be normalized or not. Normalization refers to subtracting the gravity from the signal. Defaults to False.
        modify (bool, optional): Parameter to decide if a modification is required or not. For some of the activitiy indices, 
                                 some extra modifications are required following standard preprocessing steps. These are represented as "_modified". Defaults to False.
        filter_type (str, optional): Type of the filter. Defaults to 'bandpass'.
        N (int, optional): Order of the filter. Defaults to 2.
        f1 (float, optional): Lower cutoff frequency of the filter. Defaults to 0.5.
        f2 (float, optional): Higher cutoff frequency of the filter. Defaults to 2.

    Raises:
        ValueError: If filtering_order is not given when filtering=True.
        ValueError: If filtering order is invalid.
        ValueError: If both normalize and filtering are selected as True.

    Returns:
        ArrayLike: The resulting preprocessed signal(s). The dimension can be either 1 or 3 depending on the type of dataset.
    """

    if filtering:
        if filtering_order is None:
            raise ValueError("Required parameter filtered_order.")

        elif filtering_order =='pre':
            f_x = filter_signal(accx, filter_type=filter_type, N=N, fs=sampling_rate, f1=f1, f2=f2)
            f_y = filter_signal(accy, filter_type=filter_type, N=N, fs=sampling_rate, f1=f1, f2=f2)
            f_z = filter_signal(accz, filter_type=filter_type, N=N, fs=sampling_rate, f1=f1, f2=f2)
            if magnitude:
                mag = _calc_magnitude(f_x, f_y, f_z) #FMpre
                sig = [mag]
            else:
                if modify:
                    sig = [np.abs(f_x), np.abs(f_y), np.abs(f_z)] #FXYZ_modified
                else:
                    sig = [f_x, f_y, f_z] #FXYZ

        elif filtering_order == 'post':
            mag = _calc_magnitude(accx, accy, accz)
            f_mag = filter_signal(mag, filter_type=filter_type, N=N, fs=sampling_rate, f1=f1, f2=f2) #FMpost
            sig = [f_mag]
            if modify:
                sig = [np.abs(f_mag)] #FMpost_modified       
            
        else:
            raise ValueError(f"Invalid 'filtering_order' value for `generate_dataset`: {filtering_order}. Should be one of ['pre', 'post']")
    else:
        if not magnitude:
            sig = [accx, accy, accz] #UFXYZ
        else:
            mag = _calc_magnitude(accx, accy, accz) #UFM
            if not modify and normalize:     
                sig = [mag-1] #UFNM
            elif modify and not normalize:  
                sig = [np.abs(mag - len(mag))] #UFM_modified
            elif modify and normalize:
                raise ValueError("Both normalization and modify cannot be True!")
            else:
                sig = [mag]

    return sig

def calc_threshold(sig: ArrayLike, dim: int, input_type:str) -> List:
    """Calculates the threshold level in "g", as standard deviation of the signal (It is calculated as "SD+g" for the 'UFM' dataset.)

    Args:
        sig (ArrayLike): Preprocessed acceleration signal(s).
        dim (int): Input dimension
        input_type (str): Type of dataset 

    Raises:
        ValueError: If the dimension is invalid.

    Returns:
        List: List of calculated threshold level(s).
    """

    if dim == 1:
        th = np.std(sig[0])
        if input_type =='UFM':
            threshold = [th+1]
        else:
            threshold = [th]

    elif dim == 3:
        th_x = np.std(sig[0])
        th_y = np.std(sig[1])
        th_z = np.std(sig[2])
        threshold = [th_x,th_y,th_z]
        
    else:
        raise ValueError("Invalid dimension!")
        
    return threshold

def _calc_magnitude(sig_x: ArrayLike, sig_y: ArrayLike, sig_z: ArrayLike) -> ArrayLike :
    """Calculates the magnitude signal from the axial acceleration signals.

    Args:
        sig_x (ArrayLike): Acceleration vector for the x-axis.
        sig_y (ArrayLike): Acceleration vector for the y-axis.
        sig_z (ArrayLike): Acceleration vector for the y-axis.

    Returns:
        ArrayLike: Magnitude signal.
    """
    return np.sqrt(np.square(sig_x) + np.square(sig_y) + np.square(sig_z)) #acc signals should be in "g"

def _calc_pim(sig: ArrayLike, dim: int, sampling_rate: float, triaxial:bool) -> list:
    """Calculates activity index using Proportional Integration Method (PIM).

    Args:
        sig (ArrayLike): List of acceleration signal(s).
        dim (int): Input dimension
        sampling_rate (float): Sampling rate of the acceleration signal(s).
        triaxial (bool): Parameter to decide if triaxial metrics should be combined into a single metric or not

    Raises:
        ValueError: If the dimension is invalid.

    Returns:
        list: Calculated PIM value(s).
    """
    
    if dim == 1:
        pim = [np.sum(sig[0]) / sampling_rate]
    elif dim == 3:
        pim_x = np.sum(sig[0]) / sampling_rate
        pim_y = np.sum(sig[1]) / sampling_rate
        pim_z = np.sum(sig[2]) / sampling_rate
        
        if not triaxial:
            pim= [np.sqrt(np.square(pim_x) + np.square(pim_y) + np.square(pim_z))]
        else:
            pim = [pim_x, pim_y, pim_z]
    else:
        raise ValueError("Invalid dimension!")
 
    return pim
        
def _calc_zcm(sig: ArrayLike, dim: int, threshold: float, triaxial:bool) -> list:
    """Calculates activity index using Zero Crossing Method (ZCM).

    Args:
        sig (ArrayLike): List of acceleration signal(s).
        dim (int): Input dimension
        threshold (float): Threshold level in "g".
        triaxial (bool): Parameter to decide if triaxial metrics should be combined into a single metric or not

    Raises:
        ValueError: If the threshold level is not provided.
        ValueError: If the dimension is invalid. 

    Returns:
        list: Calculated ZCM value(s).
    """
    
    if threshold is None:
        raise ValueError('Threshold value is required for this metric.')

    else:
        if dim == 1:
            zcm = 0        
            for i in range(len(sig[0])-1):
                if sig[0][i] < threshold[0] and sig[0][i+1] >= threshold[0]:
                    zcm += 1
            zcm=[zcm]
        elif dim == 3:
            zcm_x = 0
            zcm_y = 0
            zcm_z = 0
            for i in range(len(sig[0])-1):
                if sig[0][i] < threshold[0] and sig[0][i+1] >= threshold[0]:
                    zcm_x += 1
                if sig[1][i] < threshold[1] and sig[1][i+1] >= threshold[1]:
                    zcm_y += 1            
                if sig[2][i] < threshold[2] and sig[2][i+1] >= threshold[2]:
                    zcm_z += 1
            if not triaxial:
                zcm= [np.sqrt(np.square(zcm_x) + np.square(zcm_y) + np.square(zcm_z))]
            else:
                zcm = [zcm_x, zcm_y, zcm_z]
        else:
            raise ValueError("Invalid dimension!")
        
    return zcm

def _calc_tat(sig: ArrayLike, dim: int, sampling_rate: float, threshold: float, triaxial:bool) -> List:
    """Calculates activity index using Time Above Threshold Method (TAT).

    Args:
        sig (ArrayLike): List of acceleration signal(s).
        dim (int): Input dimension
        sampling_rate (float): Sampling rate of the acceleration signal(s).
        threshold (float): Threshold level in "g".
        triaxial (bool): Parameter to decide if triaxial metrics should be combined into a single metric or not

    Raises:
        ValueError: If the threshold level is not provided.
        ValueError: If the dimension is invalid. 

    Returns:
        List: Calculated TAT value(s).
    """

    if threshold is None:
        raise ValueError('Threshold value is required for this metric.')
    else:
        if dim == 1:
            tat = [len(sig[0][sig[0] >= threshold[0]]) / sampling_rate]

        elif dim == 3:
            tat_x = len(sig[0][sig[0] >= threshold[0]]) / sampling_rate
            tat_y = len(sig[1][sig[1] >= threshold[1]]) / sampling_rate
            tat_z = len(sig[2][sig[2] >= threshold[2]]) / sampling_rate
            if not triaxial:
                tat= [np.sqrt(np.square(tat_x) + np.square(tat_y) + np.square(tat_z))]
            else:
                tat = [tat_x, tat_y, tat_z]

        else:
            raise ValueError("Invalid dimension!")
        
    return tat

def _calc_mad(sig: ArrayLike, dim: int, triaxial:bool) -> List:
    """Calculates activity index using Mean Amplitude Deviation Method (MAD).

    Args:
        sig (ArrayLike): List of acceleration signal(s).
        dim (int): Input dimension
        triaxial (bool): Parameter to decide if triaxial metrics should be combined into a single metric or not

    Raises:
        ValueError: If the dimension is invalid.

    Returns:
        List: Calculated MAD value(s).
    """

    if dim == 1:
        mad = [np.sum(np.abs(sig[0] - np.mean(sig[0]))) / len(sig[0])]
    elif dim == 3:
        mad_x = np.sum(np.abs(sig[0] - np.mean(sig[0]))) / len(sig[0])
        mad_y = np.sum(np.abs(sig[1] - np.mean(sig[1]))) / len(sig[1])    
        mad_z = np.sum(np.abs(sig[2] - np.mean(sig[2]))) / len(sig[2])
        if not triaxial:
            mad= [np.sqrt(np.square(mad_x) + np.square(mad_y) + np.square(mad_z))]
        else:
            mad = [mad_x, mad_y, mad_z]
    else:
        raise ValueError("Invalid dimension!")
        
    return mad   

def _calc_enmo(sig: ArrayLike) -> List:
    """Calculates activity index using Euclidian Norm Minus One Method (ENMO).

    Args:
        sig (ArrayLike): List of acceleration signal(s).

    Returns:
        List: Calculated ENMO value(s).
    """

    enmo = [np.sum(sig[0][sig[0] >= 1]) / len(sig[0])]

    return enmo

def _calc_hfen(sig: ArrayLike, dim: int, triaxial:bool) -> List:
    """Calculates activity index using High-pass Filtered Euclidian Norm (HFEN).

    Args:
        sig (ArrayLike): List of acceleration signal(s).
        dim (int): Input dimension
        triaxial (bool): Parameter to decide if triaxial metrics should be combined into a single metric or not

    Raises:
        ValueError: If the dimension is invalid. 

    Returns:
        List: Calculated HFEN value(s).
    """

    if dim == 1:    
        hfen = [np.sum(sig[0]) / len(sig[0])]
    elif dim == 3:
        hfen_x = np.sum(sig[0]) / len(sig[0])
        hfen_y = np.sum(sig[1]) / len(sig[1])
        hfen_z = np.sum(sig[2]) / len(sig[2])
        if not triaxial:
            hfen= [np.sqrt(np.square(hfen_x) + np.square(hfen_y) + np.square(hfen_z))]
        else:
            hfen = [hfen_x, hfen_y, hfen_z]
    else:
        raise ValueError("Invalid dimension!")
        
    return hfen

def _calc_ai(sig: ArrayLike, baseline_variance: float) -> List:
    """Calculates activity index using Activity Index Method (AI).

    Args:
        sig (ArrayLike): List of acceleration signal(s).
        baseline_variance (float): Baseline variance, corresponding to the variance of acceleration signal at rest (no movement).

    Raises:
        ValueError: If the baseline variance level is not provided.

    Returns:
        List: Calculated AI value(s).
    """

    if baseline_variance is None:
        raise ValueError('Baseline variance is required for this metric.')
    else:
        x_ = np.var(sig[0]) - baseline_variance[0]
        y_ = np.var(sig[1]) - baseline_variance[1]
        z_ = np.var(sig[2]) - baseline_variance[2]
        ai = [np.sqrt(max([np.mean([x_,y_,z_]), 0]))]
    
    return ai
