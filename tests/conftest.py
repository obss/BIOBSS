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
    return np.asarray([ 48, 104, 157, 207, 257, 305, 352, 400, 452, 501, 551, 605])

@pytest.fixture(scope='package')
def ecg_Rpeaks():
    return np.asarray([94, 259, 430, 595, 758, 935, 1101, 1263, 1427, 1593, 1760, 1914, 2087, 2253, 2422])

@pytest.fixture(scope='package')
def ecg_fiducials():
    fiducials_ecg = {'ECG_P_Peaks': np.asarray([61, 225, 390, 555, 726, 894, 1061, 1223, 1388, 1552, 1720, 1882, 2047, 2214, 2382]),
                    'ECG_Q_Peaks': np.asarray([79, 247, 408, 573, 750, 915, 1077, 1238, 1406, 1571, 1735, 1900, 2065, 2226, 2401]),
                    'ECG_S_Peaks': np.asarray([102,267,438,660,766,956,1107,1305,1453,1615,1826,1922,2156,2324,2492]),
                    'ECG_T_Peaks': np.asarray([148, 316, 474, 689, 813, 981, 1146, 1361, 1475, 1639, 1855, 1966, 2184, 2332, 2524]),
                    'ECG_P_Onsets': np.asarray([51, 215, 378, 544, 715, 883, 1049, 1211, 1377, 1542, 1708, 1871, 2036, 2202, 2371]),
                    'ECG_T_Offsets': np.asarray([164, 329, 490, 696, 827, 998, 1165, 1367, 1491, 1654, 1861, 1983, 2191, 2337, 2525])}
    return fiducials_ecg

@pytest.fixture(scope='package')
def ppg_irregular():
    return np.asarray([0, 5, 60, 117, 130, 169, 219, 270, 290, 317, 364, 413, 464, 514, 563, 618, 630])

@pytest.fixture(scope='package')
def ppg_fiducials():
    fiducials_ppg = {'S_waves': np.asarray([62, 118, 170, 221, 271, 318, 365, 414, 465, 515, 564, 619]),
                    'O_waves': np.asarray([48, 104, 157, 207, 257, 305, 352, 400, 452, 501, 551, 605]),
                    'N_waves':np.asarray([72, 128, 180, 232, 281, 327, 375, 424, 475, 523, 573]),
                    'D_waves':np.asarray([77, 133, 185, 236, 285, 332, 379, 428, 479, 527, 577])}
    return fiducials_ppg

@pytest.fixture(scope='package')
def vpg_fiducials():
    fiducials_vpg = {'w_waves':np.asarray([55, 112, 164, 214, 265, 312, 359, 408, 459, 509, 558, 613]),
                    'y_waves':np.asarray([69, 123, 176, 228, 276, 323, 370, 421, 471, 520, 569]),
                    'z_waves':np.asarray([76, 132, 184, 235, 285, 331, 378, 428, 478, 527, 576])}
    return fiducials_vpg

@pytest.fixture(scope='package')
def apg_fiducials():
    fiducials_apg = {'a_waves':np.asarray([51, 109, 160, 211, 261, 309, 355, 404, 455, 505, 554, 609]),
                    'b_waves':np.asarray([58, 115, 167, 218, 268, 315, 362, 411, 462, 512, 561, 616]),
                    'c_waves':np.asarray([64, 128, 180, 232, 281, 327, 375, 424, 475, 523, 573]),                              
                    'd_waves':np.asarray([65, 128, 180, 232, 281, 327, 375, 424, 475, 523, 573]),
                    'e_waves':np.asarray([72, 128, 180, 232, 281, 327, 375, 424, 475, 523, 573])}
    return fiducials_apg