import os
import zipfile
import shutil
import csv
import datetime
import math
import time
import collections
from collections import OrderedDict
import os.path
import pandas as pd
import glob



def unzip_and_rename(zipfiles_dir,temp_dir,csvfiles_dir,zip_file_name):
    """Extracts all files from the zip archive, renames the files
    and moves into a folder named as record id. 

    Args:
        zipfiles_dir (_type_): Directory of the zip archives
        temp_dir (_type_): Directory of the temporary folder to extract files
        csvfiles_dir (_type_): Directory of the csv files
        zip_file_name (_type_): Name of the zip archive

    Returns:
        _type_: None
    """
    zip_file_path=zipfiles_dir + zip_file_name

    file_parts=os.path.splitext(zip_file_name)
    record_id=file_parts[0]

    # Current filename: New filename
    name_map = {"ACC.csv": record_id+"_ACC.csv",
                "BVP.csv": record_id+"_BVP.csv",
                "EDA.csv": record_id+"_EDA.csv",
                "HR.csv": record_id+"_HR.csv",
                "IBI.csv": record_id+"_IBI.csv",
                "tags.csv": record_id+"_tags.csv",
                "TEMP.csv": record_id+"_TEMP.csv"}


    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)


    for file in os.listdir(temp_dir):
        
        os.chdir(csvfiles_dir)
        path=os.path.join(csvfiles_dir,record_id)

        if not os.path.exists(record_id):
            os.mkdir(path)

        if file in name_map.keys():

            src=os.path.join(temp_dir,file)
            
            dest=os.path.join(path,name_map[file])        
            
            shutil.copy(src, dest)

    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))    

    return None


def readFile(file):
    """Reads a csv file, corrects for the timezone and forms the data into a dictionary.

    Args:
        file (_type_): original csv file

    Returns:
        _type_: Dictionary of the data
    """
    dict = OrderedDict()

    with open(file, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter='\n')
        i =0;
        for row in reader:
            if(i==0):
                timestamp=row[0]
                #print(timestamp)
                timestamp=float(timestamp)+3600*3 #Time Zone Correction - will need to change depending on time zone!
                #print(timestamp)
            elif(i==1):
                hertz = float(row[0])
            elif(i==2):
                dict[timestamp]=row[0]
            else:
                timestamp = timestamp + 1.0/hertz
                dict[timestamp]=row[0]
            i = i+1.0
    return dict


def formatfile(filesource, file, idd, typed):
    """Formats the data into a dataframe using ISO8601 datetime format and writes to csv.

    Args:
        file (_type_): original csv file
        idd (_type_): record id (name of the zip archive)
        typed (_type_): signal type (EDA, BVP, HR, TEMP)
    """
    EDA = {}
    EDA = readFile(file = file)
    EDA =  {datetime.datetime.utcfromtimestamp(k).strftime('%Y-%m-%d %H:%M:%S.%f'): v for k, v in EDA.items()}
    EDAdf = pd.DataFrame.from_dict(EDA, orient='index', columns=['EDA'])
    print(EDAdf)
    EDAdf['EDA'] = EDAdf['EDA'].astype(float)
    
    EDAdf['Datetime'] =EDAdf.index
    EDAdf['Datetime'] = pd.to_datetime(EDAdf['Datetime'], format='%Y-%m-%dT%H:%M:%S.%f')
    EDAdf  = EDAdf.set_index('Datetime')
    
    out_filename = (filesource + idd + '\\' + idd + '_' + typed + '_formatted.csv')
    EDAdf.to_csv(out_filename, mode='a', header=False)
    print('Done')

def importandexport(filesource, idd, typed):
    """Finds the files in the folder named as record id and runs formatfile for each file.

    Args:
        idd (_type_): record id (name of the zip archive)
        typed (_type_): signal type (EDA, BVP, HR, TEMP)
    """
    configfiles = glob.glob((filesource + idd + '\\' + idd+ '_'+ typed + '.csv'))
    print(configfiles)
    
    [formatfile(filesource, file, idd, typed) for file in configfiles]
    print(('Completed Import and Export of:' + typed))



def processAcceleration(x,y,z):
    x = float(x)
    y = float(y)
    z = float(z) 
    return {'x':x,'y':y,'z':z}


def readAccFile(file):
    dict = OrderedDict()
    
    with open(file, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        i=0;
        for row in reader:
            if(i == 0):
                timestamp = float(row[0])+3600*3 #Time Zone Correction
            elif(i == 1):    
                hertz=float(row[0])
            elif(i == 2):
                dict[timestamp]= processAcceleration(row[0],row[1],row[2])
            else:
                timestamp = timestamp + 1.0/hertz 
                dict[timestamp] = processAcceleration(row[0],row[1],row[2])
            i = i + 1
        return dict


def formatAccfile(filesource, file, idd, typed):
    EDA = {}
    EDA = readAccFile(file = file)
    EDA =  {datetime.datetime.utcfromtimestamp(k).strftime('%Y-%m-%d %H:%M:%S.%f'): v for k, v in EDA.items()}
    EDAdf = pd.DataFrame.from_dict(EDA, orient='index', columns=['x', 'y', 'z'])
    
    EDAdf['x'] = EDAdf['x'].astype(float)
    EDAdf['y'] = EDAdf['y'].astype(float)
    EDAdf['z'] = EDAdf['z'].astype(float)
    
    EDAdf['Datetime'] =EDAdf.index
    EDAdf['Datetime'] = pd.to_datetime(EDAdf['Datetime'], format='%Y-%m-%dT%H:%M:%S.%f')
    EDAdf  = EDAdf.set_index('Datetime')
    
    out_filename = (filesource + idd + '\\' + idd + '_' + typed + '_formatted.csv')
    EDAdf.to_csv(out_filename, mode='a', header=False)
    print('Done')


def importandexportAcc(filesource, idd, typed):
    configfiles = glob.glob((filesource + idd + '\\' + idd+ '_'+ typed + '.csv'))
    print(configfiles)
    
    [formatAccfile(filesource, file, idd, typed) for file in configfiles]
    print(('Completed Import and Export of:' + typed))


def importIBI(filesource, file, idd, typed):
    IBI = pd.read_csv(file, header=None)
    timestampstart = float(IBI[0][0])+3600*3
    IBI[0] = (IBI[0][1:len(IBI)]).astype(float)+timestampstart
    IBI = IBI.drop([0])
    IBI[0] = IBI[0].apply(lambda x: datetime.datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S.%f'))
    IBI  = IBI.set_index(0)
    
    out_filename = (filesource + idd + '\\' + idd + '_' + typed + '_formatted.csv')
    IBI.to_csv(out_filename, mode='a', header=False)
    print('Done')


def importandexportIBI(filesource, idd, typed):
    configfiles = glob.glob((filesource + idd + '\\' + idd+ '_'+ typed + '.csv'))
    print(configfiles)
    
    [importIBI(filesource, file, idd, typed) for file in configfiles]
    print(('Completed Import and Export of:' + typed))