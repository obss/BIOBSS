import pandas as pd
import os
import numpy as np
import pytest


@pytest.fixture(scope='package')
def load_sample_ecg():

    data_dir = os.path.join(os.path.normpath(os.getcwd() + os.sep + os.pardir),"sample_data")
    info={}
    sample_data = pd.DataFrame()

    filename='ecg_sample_data.csv'
    data=pd.read_csv(os.path.join(data_dir,filename))

    #Select the first segment to be used in the examples
    fs=256
    L=10
    sig=np.asarray(data.iloc[:fs*L,0])
    
    info['sampling_rate']=fs
    info['signal_length']=L
    sample_data['ECG']=sig

    return sample_data, info

@pytest.fixture(scope='package')
def load_sample_ppg():

    data_dir = os.path.join(os.path.normpath(os.getcwd() + os.sep + os.pardir),"sample_data")
    info={}
    sample_data = pd.DataFrame()

    filename='ppg_sample_data.csv'
    data=pd.read_csv(os.path.join(data_dir,filename), header=None)

    #Select the first segment to be used in the examples
    fs=64
    L=10
    sig=np.asarray(data.iloc[0,:])  

    info['sampling_rate']=fs
    info['signal_length']=L
    sample_data['PPG']=sig

    return sample_data, info

@pytest.fixture(scope='package')
def load_sample_acc():

    data_dir = os.path.join(os.path.normpath(os.getcwd() + os.sep + os.pardir),"sample_data")
    info={}
    sample_data = pd.DataFrame()

    filename='acc_sample_data.csv'
    data=pd.read_csv(os.path.join(data_dir,filename), header=None)

    #Select the first 60s segment to be used in the examples
    fs=32
    L=60
    accx=np.asarray(data.iloc[:fs*L,0]) #x-axis acceleration signal
    accy=np.asarray(data.iloc[:fs*L,1]) #y-axis acceleration signal
    accz=np.asarray(data.iloc[:fs*L,2]) #z-axis acceleration signal

    info['sampling_rate']=fs
    info['signal_length']=L
                
    sample_data['ACCx']=accx
    sample_data['ACCy']=accy
    sample_data['ACCz']=accz

    return sample_data, info

@pytest.fixture(scope='package')
def load_sample_eda():

    data_dir = os.path.join(os.path.normpath(os.getcwd() + os.sep + os.pardir),"sample_data")
    info={}
    sample_data = pd.DataFrame()

    filename='EDA_Chest.pkl'
    fs = 700
    L = 5920
    sig = pd.read_pickle(os.path.join(data_dir,filename))
    # Flatten the data
    sig = sig.flatten()

    info['sampling_rate']=fs
    info['signal_length']=L
    sample_data['EDA']=sig

    return sample_data, info

@pytest.fixture(scope='package')
def ppg_peaks():
    return np.asarray([60, 117, 169, 219, 270, 317, 364, 413, 464, 514, 563])

@pytest.fixture(scope='package')
def ppg_onsets():
    return np.asarray([ 49, 105, 159, 209, 259, 306, 355, 402, 453, 503, 552, 606])

@pytest.fixture(scope='package')
def ecg_Rpeaks():
    return np.asarray([94, 259, 430, 595, 758, 935, 1101, 1263, 1427, 1593, 1760, 1914, 2087, 2253, 2422])

@pytest.fixture(scope='package')
def ecg_fiducials():
    fiducials = {'ECG_P_Peaks': [61, 225, 390, 555, 726, 894, 1061, 1223, 1388, 1552, 1720, 1882, 2047, 2214, 2382],
                    'ECG_Q_Peaks': [79, 247, 408, 573, 750, 915, 1077, 1238, 1406, 1571, 1735, 1900, 2065, 2226, 2401],
                    'ECG_S_Peaks': [102,267,438,660,766,956,1107,1305,1453,1615,1826,1922,2156,2324,2492],
                    'ECG_T_Peaks': [148, 316, 474, 689, 813, 981, 1146, 1361, 1475, 1639, 1855, 1966, 2184, 2332, 2524],
                    'ECG_P_Onsets': [51, 215, 378, 544, 715, 883, 1049, 1211, 1377, 1542, 1708, 1871, 2036, 2202, 2371],
                    'ECG_T_Offsets': [164, 329, 490, 696, 827, 998, 1165, 1367, 1491, 1654, 1861, 1983, 2191, 2337, 2525]
    }
    return fiducials

@pytest.fixture(scope='package')
def ppg_irregular():
    return np.asarray([0, 5, 60, 117, 130, 169, 219, 270, 290, 317, 364, 413, 464, 514, 563, 618, 630])
