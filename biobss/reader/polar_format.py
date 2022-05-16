import os
import glob
from time import time
import pandas as pd
import datetime
import numpy as np
from scipy import interpolate
from scipy import signal
from numpy.typing import ArrayLike

from biobss.reader import polar_reader

COLUMN_NAMES={
    'PPG' : ['channel 0','channel 1', 'channel 2', 'ambient'],
    'ACC' : ['X [mg]','Y [mg]', 'Z [mg]'],
    'MAGN' : ['X [G]','Y [G]', 'Z [G]'],
    'GYRO' : ['X [dps]','Y [dps]', 'Z [dps]'],
    }


def txt_to_csv(txt_dir:str, file_types=['HR','PPI','ACC','PPG','MAGN','GYRO'], record_id: str=None):
    """Reads txt files and saves as csv files.

    Args:
        txt_dir (str): Directory of the text files.
        file_types (list, optional): File types. Defaults to ['HR','PPI','ACC','PPG','MAGN','GYRO'].
        record_id (str, optional): Record id of the file. It is used to rename the file. Defaults to None.
    """
    for root, _, files in os.walk(txt_dir):

        if len(files)!=0:
            csv_dir=root.replace("txt_files","csv_files")
            os.chdir(root)

            if not os.path.exists(csv_dir):
                os.makedirs(csv_dir)

                marker_files=glob.glob("*MARKER*.txt*")
                if marker_files:
                    marker_file=marker_files[0]

                    txt_path=root+"\\"+marker_file
                    if record_id is None:
                        csv_path=csv_dir+"\\"+os.path.splitext(marker_file)[0]+'.csv' 
                    else:
                        csv_path=csv_dir+"\\"+record_id+"_marker"+'.csv'
                        
                    content_df=pd.read_csv(txt_path, delimiter=';')
                    content_df.to_csv(csv_path,index=None)

                for file_type in file_types:

                    filenames=glob.glob("*_{}.txt*".format(file_type))
                    if filenames:
                        filename=filenames[0]

                        txt_path=root+"\\"+filename
                        if record_id is None:
                            csv_path=csv_dir+"\\"+os.path.splitext(filename)[0]+'.csv' 
                        else:
                            csv_path=csv_dir+"\\"+record_id+"_"+file_type+'.csv'

                        if file_type == 'PPG':
                            ppg_header=pd.read_csv(txt_path, delimiter=';',nrows=1).columns.tolist()
                            content_df=pd.read_csv(txt_path, delimiter=';')
                            content_df.drop(content_df.columns[-1], axis=1, inplace=True)
                            content_df.reset_index(level=0,inplace=True)
                            content_df.columns=ppg_header
                            content_df.to_csv(csv_path,index=None)

                        else:
                            content_df=pd.read_csv(txt_path, delimiter=';')
                            content_df.to_csv(csv_path,index=None)


def timestamp_to_msec(timestamp_df:pd.DataFrame, start_time:datetime.datetime=None) -> ArrayLike:
    """Converts timestamp to time in milliseconds. 

    Args:
        timestamp_df (pd.DataFrame): Timestamps
        start_time (datetime.datetime, optional): Reference starting time. Defaults to None. If a specific value is given, all time points are calculated referenced to it.

    Returns:
        ArrayLike: Array of timepoints in msec.
    """

    if start_time is None:

        timestamps=timestamp_df.values.tolist()
        tses = [datetime.datetime.fromisoformat(x) for x in timestamps]
        timediff=[y-x for x,y in zip(tses,tses[1:])]
        timediff_usec=[t.days/86400*1000000+t.seconds*1000000+t.microseconds for t in timediff]
        timediff_usec.insert(0,0)
        timediff_usec= np.cumsum(np.asarray(timediff_usec))

    else:

        timestamps=timestamp_df.values.tolist()
        tses = [datetime.datetime.fromisoformat(x) for x in timestamps]
        timediff=[y-start_time for y in tses]
        timediff_usec=[t.days/86400*1000000+t.seconds*1000000+t.microseconds for t in timediff]


    return np.asarray(timediff_usec)/1000


def get_start_time(csv_dir:str, file_types:list=['HR','PPI','ACC','PPG','MAGN','GYRO'],marker:bool=False) ->datetime.datetime:
    """Returns the earliest timestamp for selected filetypes.

    Args:
        csv_dir (str): Directory of the csv files for a specific record.
        file_types (list, optional): File types to be included. Defaults to ['HR','PPI','ACC','PPG','MAGN','GYRO'].
        marker (bool, optional): True if marker file exists. Defaults to False.

    Returns:
        datetime.datetime: Datetime object for start time.
    """
    files=os.listdir(csv_dir)
    os.chdir(csv_dir)
    if len(files)!=0:
        csv_files=[]
        sig_start=[]
        sig_stop=[]
        
        if marker:

            marker_files=glob.glob("*MARKER*.csv*")
            if marker_files:
                csv_files.append(marker_files[0])
                
        for file_type in file_types:
            filenames=glob.glob("*_{}.csv*".format(file_type))
            if filenames:
                csv_files.append(filenames[0])
   
        
        for filename in csv_files:

            filepath=csv_dir+"\\"+filename    
            df= pd.read_csv(filepath)
            timestamps=df['Phone timestamp'].values.tolist()
            if timestamps:
                tses = [datetime.datetime.fromisoformat(x) for x in timestamps]
                sig_start.append(tses[0])
                sig_stop.append(tses[-1])

        sorted_start=sorted(sig_start)
        t_start=sorted_start[0]

    return t_start


def update_csv(csv_dir:str, file_types:list=['HR','PPI','ACC','PPG','MAGN','GYRO'],marker:bool=False,start_time=None):
    """Adds a column to the selected csv files for relative time in milliseconds. The timepoints are calculated referenced to the start_time.

    Args:
        csv_dir (str): Directory of the csv files for a specific record.
        file_types (list, optional): File types to be included. Defaults to ['HR','PPI','ACC','PPG','MAGN','GYRO'].
        marker (bool, optional): True if marker file exists. Defaults to False.
        start_time (datetime.datetime, optional): Reference start time. Defaults to None. If None, it is calculated as the earliest timestamp of the selected files.
    """

    files=os.listdir(csv_dir)
    os.chdir(csv_dir)

    if len(files)!=0:

        if start_time is None:

            start_time=get_start_time(csv_dir,file_types=file_types,marker=marker)

        csv_files=[]

        if marker:

            marker_files=glob.glob("*MARKER*.csv*")
            if marker_files:
                csv_files.append(marker_files[0])

        for file_type in file_types:
            filenames=glob.glob("*_{}.csv*".format(file_type))
            if filenames:
                csv_files.append(filenames[0])            


        for filename in csv_files:

            filepath=csv_dir+"\\"+filename
            df= pd.read_csv(filepath)
            timediff_usec=timestamp_to_msec(df['Phone timestamp'],start_time)
            df['Time_record (ms)']=timediff_usec
            df.to_csv(filepath,index=None)


def calculate_sync_time(csv_dir:str, time_step:float,file_types=['HR','PPI','ACC','PPG','MAGN','GYRO'], save_file:bool=False, marker=False) -> pd.DataFrame:
    """Generates a time list for synchronization.

    Args:
        csv_dir (str): Directory of the csv files for a specific record.
        time_step (float): Time step in milliseconds.
        file_types (list, optional): File types to be included. Defaults to ['HR','PPI','ACC','PPG','MAGN','GYRO'].
        save_file (bool, optional): If True, the generated time list is saved as a txt file. Defaults to False.
        marker (bool, optional): True if marker file exists. Defaults to False.

    Raises:
        ValueError: If the directory is empty.

    Returns:
        pd.DataFrame: Dataframe of generated time list.
    """

    os.chdir(csv_dir)
    files = os.listdir(csv_dir)

    if len(files)==0:
        raise ValueError("Empty directory")

    else:
        csv_files=[]
        sig_start=[]
        sig_stop=[]
        
        if marker:

            marker_files=glob.glob("*MARKER*.csv*")
            if marker_files:
                csv_files.append(marker_files[0])
                
        for file_type in file_types:
            filenames=glob.glob("*_{}.csv*".format(file_type))
            if filenames:
                csv_files.append(filenames[0]) 
                
        for filename in csv_files:

            filepath=csv_dir+"\\"+filename
            df= pd.read_csv(filepath)
            tses=df['Time_record (ms)'].values.tolist()
            if tses:
                sig_start.append(tses[0])
                sig_stop.append(tses[-1])

        sorted_start=sorted(sig_start)
        t_start=sorted_start[-1]

        sorted_stop=sorted(sig_stop)
        t_stop=sorted_stop[0]

        time_list=np.arange(t_start,t_stop,time_step)
        timelist_df=pd.DataFrame(time_list)

        if save_file:    
            timelist_df.to_csv(csv_dir+"\\times.txt",header=None,index=None)
        
    return timelist_df



def synchronize_signals(csv_dir:str, time_list=None, interp_method:str='linear', sampling_rate:int=1000, resampling_rate:int=1000, file_types:list=['ACC','PPG','MAGN','GYRO'],save_files:bool=False) -> pd.DataFrame:
    """Synchronizes the signals by interpolating for the time_list.

    Args:
        csv_dir (str): Directory of the csv files.
        time_list (_type_, optional): Time list. If it is not given, it is read from the file. Defaults to None.
        interp_method (str, optional): Interpolation method. Defaults to 'Linear'.
        sampling_rate (int, optional): Sampling rate. Defaults to 1000.
        resampling_rate (int, optional): Resampling rate. Defaults to 1000.
        file_types (list, optional): Signal types to be synchronized. Defaults to ['ACC','PPG','MAGN','GYRO'].
        save_files (bool, optional): If True, the synchronized signals are saved as a csv file. Defaults to False.

    Raises:
        ValueError: If no file for the given file_type.

    Returns:
        pd.DataFrame: DataFrame including synchronized signals and time points.
    """
    
    os.chdir(csv_dir)
    data=pd.DataFrame()
    out_file="sync"

    if time_list is None:
        time_list=pd.read_csv('times.txt',header=None).values.tolist()


    for file_type in file_types:

        filenames=glob.glob("*_{}.csv*".format(file_type))
        if filenames:
            out_file=out_file+"_"+file_type
            filename=filenames[0]
            filepath=csv_dir+"\\"+filename
            df= pd.read_csv(filepath)
            tses=df['Time_record (ms)'].values.tolist()

            columns= COLUMN_NAMES[file_type]
            
            for column_name in columns:
                sig=df[column_name].values.tolist()
                interp=interpolate.interp1d(tses,sig,kind=interp_method)
                sig_new=np.asarray(interp(time_list))

                #biobss resample module will be used here.
                ratio=(resampling_rate/sampling_rate)
                target_length = round (len(sig_new) * ratio)
                resampled_sig,resampled_t=signal.resample(sig_new,num=target_length,t=time_list.values)
                data[column_name]=pd.Series(np.squeeze(resampled_sig))
            
        else:
            raise ValueError("No file for filetype:", file_type)

    data.insert(0,'Time_record (ms)',pd.Series(resampled_t))

    if save_files:
        data.to_csv(csv_dir+"\\"+out_file+".csv",index=None)

    return data


def segment_events(filepath:str, markerpath:str,events:list,out_path:str,save_file:bool=False) -> pd.DataFrame:
    """Segments signals for events using the marker file.

    Args:
        filepath (str): Full path for the csv file to be segmented.
        markerpath (str): Full path for the marker file.
        events (list): List of events
        out_path (str): Full path for the output file.
        save_file (bool, optional): If True, saves the array of events. Defaults to False.

    Returns:
        pd.DataFrame: Dataframe of signals and corresponding events.
    """
    

    data=pd.read_csv(filepath)
    timestamps=data['Time_record (ms)']
    marked_times=pd.read_csv(markerpath)['Time_record (ms)'].values.tolist()

    event_times={}
    event_list=[]
    i=0
    for event in events:
        event_times[event] = marked_times[i:i+2]
        i=i+2
        event_start=event_times[event][0]
        event_stop=event_times[event][1]

        for timestamp in timestamps:
            if timestamp>=event_start and timestamp<=event_stop:
                event_list.append(event)

    if save_file:
        pd.DataFrame(event_list).to_csv(out_path,header=None)

    else:
        data['Events']=event_list
        return data


def csv_to_pkl(csv_dir:str,file_types:list=['HR','PPI','ACC','PPG','MAGN','GYRO'], record_id: str=None):
    """Reads csv files and saves as pkl files.

    Args:
        csv_dir (str): Directory of the csv files.
        file_types (list, optional): File types to be included. Defaults to ['HR','PPI','ACC','PPG','MAGN','GYRO'].
        record_id (str, optional): Record id of the file. It is used to rename the file. Defaults to None.
    """
    
    for root, _, files in os.walk(csv_dir):

        if len(files)!=0:
            pkl_dir=root.replace("csv_files","pkl_files")
            os.chdir(root)


            if not os.path.exists(pkl_dir):

                os.makedirs(pkl_dir)

                marker_files=glob.glob("*MARKER*.csv*") 

                if marker_files:
                    marker_file=marker_files[0]
                    csv_path=root+"\\"+marker_file 

                    if record_id is None:
                        pkl_path=pkl_dir+"\\"+os.path.splitext(marker_file)[0]+'.pkl' 
                        
                    else:
                        pkl_path=pkl_dir+"\\"+record_id+"_marker"+'.pkl'

                    info=polar_reader.polar_csv_reader(csv_path,signal_type='MARKER')

                    os.chdir(pkl_dir)
                    pd.to_pickle(info,pkl_path)
                
                for file_type in file_types:
                    os.chdir(root)
                    filenames=glob.glob("*_{}.csv*".format(file_type))

                    if filenames:
                        filename=filenames[0]
                        csv_path=root+"\\"+filename

                        if record_id is None:
                            pkl_path=pkl_dir+"\\"+os.path.splitext(filename)[0]+'.pkl' 
                        else:
                            pkl_path=pkl_dir+"\\"+record_id+"_"+file_type+'.pkl'

                        info=polar_reader.polar_csv_reader(csv_path,signal_type=file_type)
                        pd.to_pickle(info,pkl_path) 







