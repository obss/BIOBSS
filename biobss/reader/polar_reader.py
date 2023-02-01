import pandas as pd

from biobss.reader import polar_format


def _read_hr_file(filepath: str) -> pd.DataFrame:
    """Reads HR file and returns a dataframe.

    Args:
        filepath (str): Full path of the HR file (csv).

    Returns:
        pd.DataFrame: Dataframe of the signals in the file.
    """
    data = {}
    df = pd.read_csv(filepath)
    time_msec = polar_format.timestamp_to_msec(df["Phone timestamp"])

    data["Timestamp"] = df["Phone timestamp"].values.tolist()
    data["Time_sensor (ms)"] = time_msec
    data["Time_record (ms)"] = df["Time_record (ms)"].values.tolist()
    data["HR"] = df["HR [bpm]"].values.tolist()

    return data


def _read_ppi_file(filepath: str) -> pd.DataFrame:
    """Reads PPI file and returns a dataframe.

    Args:
        filepath (str): Full path of the PPI file (csv).

    Returns:
        pd.DataFrame: Dataframe of the signals in the file.
    """
    data = {}
    df = pd.read_csv(filepath)
    time_msec = polar_format.timestamp_to_msec(df["Phone timestamp"])

    data["Timestamp"] = df["Phone timestamp"].values.tolist()
    data["Time_sensor (ms)"] = time_msec
    data["Time_record (ms)"] = df["Time_record (ms)"].values.tolist()
    data["PP interval"] = df["PP-interval [ms]"].values.tolist()
    data["Error"] = df["error estimate [ms]"].values.tolist()
    data["Blocker"] = df["blocker"].values.tolist()
    data["Contact"] = df["contact"].values.tolist()
    data["Contact.1"] = df["contact.1"].values.tolist()
    data["HR"] = df["hr [bpm]"].values.tolist()

    return data


def _read_ppg_file(filepath: str) -> pd.DataFrame:
    """Reads PPG file and returns a dataframe.

    Args:
        filepath (str): Full path of the PPG file (csv).

    Returns:
        pd.DataFrame: Dataframe of the signals in the file.
    """
    data = {}
    df = pd.read_csv(filepath)
    time_msec = polar_format.timestamp_to_msec(df["Phone timestamp"])

    data["Timestamp"] = df["Phone timestamp"].values.tolist()
    data["Time_sensor (ms)"] = time_msec
    data["Time_record (ms)"] = df["Time_record (ms)"].values.tolist()
    data["PPG_Ch0"] = df["channel 0"].values.tolist()
    data["PPG_Ch1"] = df["channel 1"].values.tolist()
    data["PPG_Ch2"] = df["channel 2"].values.tolist()
    data["Ambient"] = df["ambient"].values.tolist()

    return data


def _read_acc_file(filepath: str) -> pd.DataFrame:
    """Reads ACC file and returns a dataframe.

    Args:
        filepath (str): Full path of the ACC file (csv).

    Returns:
        pd.DataFrame: Dataframe of the signals in the file.
    """
    data = {}
    df = pd.read_csv(filepath)
    time_msec = polar_format.timestamp_to_msec(df["Phone timestamp"])

    data["Timestamp"] = df["Phone timestamp"].values.tolist()
    data["Time_sensor (ms)"] = time_msec
    data["Time_record (ms)"] = df["Time_record (ms)"].values.tolist()
    data["Acc_x"] = df["X [mg]"].values.tolist()
    data["Acc_y"] = df["Y [mg]"].values.tolist()
    data["Acc_z"] = df["Z [mg]"].values.tolist()

    return data


def _read_magn_file(filepath: str) -> pd.DataFrame:
    """Reads MAGN file and returns a dataframe.

    Args:
        filepath (str): Full path of the MAGN file (csv).

    Returns:
        pd.DataFrame: Dataframe of the signals in the file.
    """
    data = {}
    df = pd.read_csv(filepath)
    time_msec = polar_format.timestamp_to_msec(df["Phone timestamp"])

    data["Timestamp"] = df["Phone timestamp"].values.tolist()
    data["Time_sensor (ms)"] = time_msec
    data["Time_record (ms)"] = df["Time_record (ms)"].values.tolist()
    data["Magn_x"] = df["X [G]"].values.tolist()
    data["Magn_y"] = df["Y [G]"].values.tolist()
    data["Magn_z"] = df["Z [G]"].values.tolist()

    return data


def _read_gyro_file(filepath: str) -> pd.DataFrame:
    """Reads GYRO file and returns a dataframe.

    Args:
        filepath (str): Full path of the GYRO file (csv).

    Returns:
        pd.DataFrame: Dataframe of the signals in the file.
    """
    data = {}
    df = pd.read_csv(filepath)
    time_msec = polar_format.timestamp_to_msec(df["Phone timestamp"])

    data["Timestamp"] = df["Phone timestamp"].values.tolist()
    data["Time_sensor (ms)"] = time_msec
    data["Time_record (ms)"] = df["Time_record (ms)"].values.tolist()
    data["Gyro_x"] = df["X [dps]"].values.tolist()
    data["Gyro_y"] = df["Y [dps]"].values.tolist()
    data["Gyro_z"] = df["Z [dps]"].values.tolist()

    return data


def _read_marker_file(filepath: str) -> pd.DataFrame:
    """Reads MARKER file and returns a dataframe.

    Args:
        filepath (str): Full path of the MARKER file (csv).

    Returns:
        pd.DataFrame: Dataframe of the signals in the file.
    """
    data = {}
    df = pd.read_csv(filepath)
    time_msec = polar_format.timestamp_to_msec(df["Phone timestamp"])

    data["Timestamp"] = df["Phone timestamp"].values.tolist()
    data["Time_sensor (ms)"] = time_msec
    data["Time_record (ms)"] = df["Time_record (ms)"].values.tolist()
    data["Start/Stop"] = df["Marker start/stop"].values.tolist()

    return data


READER_FUNCTIONS = {
    "HR": _read_hr_file,
    "PPI": _read_ppi_file,
    "PPG": _read_ppg_file,
    "ACC": _read_acc_file,
    "MAGN": _read_magn_file,
    "GYRO": _read_gyro_file,
    "MARKER": _read_marker_file,
}


def polar_csv_reader(filepath: str, signal_type: str) -> dict:
    """Reads a csv file and returns a dictionary.

    Args:
        filepath (str): Directory of the csv file.
        signal_type (str): Signal type to be processed.

    Raises:
        ValueError: If signal_type is not one of "valid_types".

    Returns:
        dict: Dictionary of signals.
    """

    valid_types = ["HR", "PPI", "ACC", "PPG", "MAGN", "GYRO", "MARKER"]

    if signal_type in valid_types:
        info = READER_FUNCTIONS[signal_type](filepath)

    else:
        raise ValueError(f"Signal type should be provided as one of {valid_types}.")

    return info
