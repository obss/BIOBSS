import os

import numpy as np
import pandas as pd

from biobss import FIXTURES_ROOT

from .file import download_from_url


def load_sample_data(data_type: str, download=True) -> tuple:
    """Loads sample data file for the given data type.

    Args:
        data_type (str): Data type. It can be one of ['PPG', 'ECG', 'ACC', 'EDA'].

    Raises:
        ValueError: If data_type is not valid.

    Returns:
        tuple: Dataframe of sample data, dictionary of information on sample data
    """

    data_type = data_type.upper()

    info = {}
    sample_data = pd.DataFrame()

    if data_type == "PPG_SHORT":
        sample_data, info = _load_sample_ppg(FIXTURES_ROOT, download)

    elif data_type == "PPG_LONG":
        sample_data, info = _load_sample_ppg_long(FIXTURES_ROOT, download)

    elif data_type == "ECG":
        sample_data, info = _load_sample_ecg(FIXTURES_ROOT, download)

    elif data_type == "ACC":
        sample_data, info = _load_sample_acc(FIXTURES_ROOT, download)

    elif data_type == "EDA":
        sample_data, info = _load_sample_eda(FIXTURES_ROOT, download)

    else:
        raise ValueError(f"Sample data does not exist for the selected data type {data_type}.")

    return sample_data, info


def _load_sample_ppg(data_dir: str, download=False) -> tuple:
    info = {}
    sample_data = pd.DataFrame()
    url = "https://raw.githubusercontent.com/obss/BIOBSS/2bce481e94ecf08bb3b78122b4ede678e447dde4/sample_data/ppg_sample_data.csv"
    filename = data_dir / "ppg_sample_data.csv"
    if not os.path.exists(filename):
        filename = os.getcwd() + "/sample_data/ppg_sample_data.csv"
        download_from_url(url, filename)
        print("Downloaded sample data to " + filename)
    data = pd.read_csv(filename, header=None)

    # Select the first segment to be used in the examples
    fs = 64
    L = 10
    sig = np.asarray(data.iloc[0, :])

    info["sampling_rate"] = fs
    info["signal_length"] = L
    sample_data["PPG"] = sig

    return sample_data, info


def _load_sample_ppg_long(data_dir: str, download=False) -> tuple:
    info = {}
    sample_data = pd.DataFrame()
    url = "https://raw.githubusercontent.com/obss/BIOBSS/main/sample_data/ppg_sample_data_long.csv"
    filename = data_dir / "ppg_sample_data_long.csv"
    if not os.path.exists(filename):
        filename = os.getcwd() + "/sample_data/ppg_sample_data_long.csv"
        download_from_url(url, filename)
        print("Downloaded sample data to " + filename)
    data = pd.read_csv(filename, header=None)
    # Select the first segment to be used in the examples
    fs = 64
    L = 40
    sig = np.asarray(data.iloc[0, :])

    info["sampling_rate"] = fs
    info["signal_length"] = L
    sample_data["PPG"] = sig

    return sample_data, info


def _load_sample_ecg(data_dir: str, download=False) -> tuple:
    info = {}
    sample_data = pd.DataFrame()
    url = "https://raw.githubusercontent.com/obss/BIOBSS/main/sample_data/ecg_sample_data.csv"
    filename = data_dir / "/sample_data/ecg_sample_data.csv"
    if not os.path.exists(filename):
        filename = os.getcwd() + "/sample_data/ecg_sample_data.csv"
        download_from_url(url, filename)
        print("Downloaded sample data to " + filename)
    data = pd.read_csv(filename, header=None)
    # Select the first segment to be used in the examples
    fs = 256
    L = 10
    sig = np.asarray(data.iloc[: fs * L, 0])

    info["sampling_rate"] = fs
    info["signal_length"] = L
    sample_data["ECG"] = sig

    return sample_data, info


def _load_sample_acc(data_dir: str, download=False) -> tuple:
    info = {}
    sample_data = pd.DataFrame()
    url = "https://raw.githubusercontent.com/obss/BIOBSS/main/sample_data/acc_sample_data.csv"
    filename = data_dir / "acc_sample_data.csv"
    if not os.path.exists(filename):
        if download:
            filename = os.getcwd() + "/sample_data/acc_sample_data.csv"
            download_from_url(url, filename)
            print("Downloaded sample data to " + filename)
    # Select the first 60s segment to be used in the examples
    data = pd.read_csv(filename, header=None)
    fs = 32
    L = 60
    accx = np.asarray(data.iloc[: fs * L, 0])  # x-axis acceleration signal
    accy = np.asarray(data.iloc[: fs * L, 1])  # y-axis acceleration signal
    accz = np.asarray(data.iloc[: fs * L, 2])  # z-axis acceleration signal

    info["sampling_rate"] = fs
    info["signal_length"] = L

    sample_data["ACCx"] = accx
    sample_data["ACCy"] = accy
    sample_data["ACCz"] = accz

    return sample_data, info


def _load_sample_eda(data_dir: str, download=False) -> tuple:
    info = {}
    sample_data = pd.DataFrame()
    url = "https://github.com/obss/BIOBSS/raw/main/sample_data/EDA_Chest.pkl"
    filename = data_dir / "EDA_Chest.pkl"
    if not os.path.exists(filename):
        if download:
            filename = os.getcwd() + "/sample_data/EDA_Chest.pkl"
            download_from_url(url, filename)
            print("Downloaded sample data to " + filename)
    fs = 700
    L = 5920
    sig = pd.read_pickle(filename)
    # Flatten the data
    sig = sig.flatten()

    info["sampling_rate"] = fs
    info["signal_length"] = L
    sample_data["EDA"] = sig

    return sample_data, info
