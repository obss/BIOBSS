import os
import pandas as pd
import numpy as np

def load_sample_data(data_type:str) -> tuple:
    """Loads sample data file for the given data type.

    Args:
        data_type (str): Data type. It can be one of ['PPG', 'ECG', 'ACC', 'EDA'].

    Raises:
        ValueError: If data_type is not valid.

    Returns:
        tuple: Dataframe of sample data, dictionary of information on sample data
    """

    data_type = data_type.upper()
    data_dir = os.path.join(os.path.normpath(os.getcwd() + os.sep + os.pardir),"sample_data")
    info={}
    sample_data = pd.DataFrame()

    if data_type == 'PPG':

        filename='ppg_sample_data.csv'
        data=pd.read_csv(os.path.join(data_dir,filename), header=None)

        #Select the first segment to be used in the examples
        fs=64
        L=10
        sig=np.asarray(data.iloc[0,:])  

        info['sampling_rate']=fs
        info['signal_length']=L
        sample_data['PPG']=sig

    elif data_type == 'ECG':

        filename='ecg_sample_data.csv'
        data=pd.read_csv(os.path.join(data_dir,filename))

        #Select the first segment to be used in the examples
        fs=256
        L=10
        sig=np.asarray(data.iloc[:fs*L,0])
        
        info['sampling_rate']=fs
        info['signal_length']=L
        sample_data['ECG']=sig

    elif data_type == 'ACC':

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

    elif data_type == 'EDA':

        filename='EDA_Chest.pkl'
        fs = 700
        L = 5920
        sig = pd.read_pickle(os.path.join(data_dir,filename))
        # Flatten the data
        sig = sig.flatten()

        info['sampling_rate']=fs
        info['signal_length']=L
        sample_data['EDA']=sig

    else:
        raise ValueError(f"Sample data does not exist for the selected data type {data_type}.")

    return sample_data, info