import csv
from typing import Tuple
import pandas as pd


def get_parser_parameters(csv_filepath: str) -> Tuple:
    """Reads the raw csv file exported using HealthManager app and returns the required parameters to parse the file.

    Args:
        csv_filepath (str): Path of the csv file.

    Returns:
        Tuple: numrow: number of rows to parse, first row: index of the first row to parse, columns: column names
    """
    numrow=0
    with open(csv_filepath, newline='') as csvfile:
        data=csv.reader(csvfile, delimiter=';')
        for idx, row in enumerate(data):
            numrow+=1
            if len(row)!=0 and row[0]=='Date':
                first_row = idx
                columns = [row[0],row[1],row[3],row[4],row[5],row[6]]
    return numrow, first_row, columns

def csv_to_df(csv_filepath: str, numrow: int, first_row: int, columns: list) -> pd.DataFrame:
    """Parses the raw csv file and returns a dataframe with the required information only.

    Args:
        csv_filepath (str): Path of the csv file.
        numrow (int): Number of rows to parse
        first_row (int): Index of the first row to parse
        columns (list): Column names

    Returns:
        pd.DataFrame: A dataframe of the blood pressure measurements.
    """
    data=[]
    with open(csv_filepath, newline='') as csvfile:
        reader=csv.reader(csvfile, delimiter=';')
        for idx,row in enumerate(reader):
            if (idx > first_row) and (idx < numrow-1) and (len(row)!=0):
                data.append([row[0],row[1],row[3],row[4],row[5],row[6]])
    data_df = pd.DataFrame(data[::-1])
    data_df.columns=columns
    return data_df