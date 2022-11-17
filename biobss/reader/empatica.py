#copied from flirt

from typing import List
import pandas as pd

#from ..util import io

__all__ = [
    "read_acc_file_into_df",
    "read_bvp_file_into_df",
    "read_eda_file_into_df",
    "read_hr_file_into_df",
    "read_ibi_file_into_df",
    "read_temp_file_into_df",
]


def read_hr_file_into_df(filepath_or_buffer) -> pd.DataFrame:
    return __read_frequency_based_file_into_df(filepath_or_buffer, ['hr'])

def read_eda_file_into_df(filepath_or_buffer) -> pd.DataFrame:
    return __read_frequency_based_file_into_df(filepath_or_buffer, ['eda'])

def read_bvp_file_into_df(filepath_or_buffer) -> pd.DataFrame:
    return __read_frequency_based_file_into_df(filepath_or_buffer, ['bvp'])

def read_temp_file_into_df(filepath_or_buffer) -> pd.DataFrame:
    return __read_frequency_based_file_into_df(filepath_or_buffer, ['temp'])

def read_acc_file_into_df(filepath_or_buffer) -> pd.DataFrame:
    return __read_frequency_based_file_into_df(filepath_or_buffer, ['acc_x', 'acc_y', 'acc_z'])

def __read_frequency_based_file_into_df(filepath_or_buffer, column_names: List[str]) -> pd.DataFrame:
    if io.is_file_like(filepath_or_buffer):
        file_to_read = filepath_or_buffer
        close_file = False
    elif isinstance(filepath_or_buffer, str):
        file_to_read = open(filepath_or_buffer, 'br')
        close_file = True
    else:
        raise ValueError('Illegal input type: %d' % type(filepath_or_buffer))

    initial_pos = file_to_read.tell()
    timestamp = pd.to_datetime(float(str(file_to_read.readline(), 'utf-8').strip().split(',')[0]), unit='s', utc=True)
    frequency = float(str(file_to_read.readline(), 'utf-8').strip().split(',')[0])
    file_to_read.seek(initial_pos)

    data = pd.read_csv(file_to_read, skiprows=2, names=column_names, index_col=False)
    data.index = pd.date_range(start=timestamp, periods=len(data), freq=str(1 / frequency * 1000) + 'ms',
                               name='datetime', tz='UTC')
    data.sort_index(inplace=True)

    if close_file:
        file_to_read.close()
    return data


def read_ibi_file_into_df(filepath_or_buffer) -> pd.DataFrame:
    """
    Reads an Empatica IBI file into a DataFrame.

    Parameters
    ----------
    filepath_or_buffer
        filepath as string or buffer (file)

    Returns
    -------
    IBIs : pd.DataFrame
        a pd.DataFrame with an 'ibi' columns, IBIs in milliseconds

    """

    if io.is_file_like(filepath_or_buffer):
        file_to_read = filepath_or_buffer
        close_file = False
    elif isinstance(filepath_or_buffer, str):
        file_to_read = open(filepath_or_buffer, 'br')
        close_file = True
    else:
        raise ValueError('Illegal input type: %d' % type(filepath_or_buffer))

    initial_pos = file_to_read.tell()
    timestamp = float(str(file_to_read.readline(), 'utf-8').strip().split(',')[0])
    file_to_read.seek(initial_pos)

    ibi = pd.read_csv(file_to_read, skiprows=1, names=['ibi'], index_col=0)
    ibi['ibi'] *= 1000  # to get ms

    ibi.index = pd.to_datetime((ibi.index * 1000 + timestamp * 1000).map(int), unit='ms', utc=True)
    ibi.index.name = 'datetime'

    if close_file:
        file_to_read.close()
    return ibi