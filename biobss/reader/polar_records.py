import os
import yaml

def add_record(yml_dir: str, yml_filename: str, record_id: str, file_types: list, marker: bool, events: list, bp: bool, fs_hr: float=None, fs_ppg: float=None, fs_acc: float=None, fs_magn: float=None, fs_gyro: float=None):
    """Adds a new record to a yml file.

    Args:
        yml_dir (str): Directory of the yml file.
        yml_filename (str): Name of the yml file.
        record_id (str): Record id.
        file_types (list): File types included in the record.
        marker (bool): True if marker file exists.
        events (list): Event list of the record.
        bp (bool): True if blood pressure measurement exists.
        fs_ppg (float, optional): Sampling rate of the PPG signal. Defaults to None.
        fs_hr (float, optional): Sampling rate of the HR signal. Defaults to None.
        fs_acc (float, optional): Sampling rate of the ACC signal. Defaults to None.
        fs_magn (float, optional): Sampling rate of the MAGN signal. Defaults to None.
        fs_gyro (float, optional): Sampling rate of the GYRO signal. Defaults to None.
    """
    if fs_ppg <= 0 or fs_hr <= 0 or fs_acc <= 0 or fs_magn <= 0 or fs_gyro <= 0: 
        raise ValueError("Sampling rate must be greater than 0. ")

    os.chdir(yml_dir)   
    info=_load_yaml(yml_filename)
        
    records=info['records']
    records[record_id]={'file_types':file_types,'marker':marker,'events':events,'bp':bp,'fs_ppg':fs_ppg,'fs_hr':fs_hr,'fs_acc':fs_acc,'fs_magn':fs_magn,'fs_gyro':fs_gyro}
    info['records']=records
 
    _save_yaml(yml_filename,info)


def update_excluded(yml_dir: str, yml_filename: str, record_id: str, add: bool=False, remove: bool=False):
    """Updates the list of excluded records in the yml file.

    Args:
        yml_dir (str): Directory of the yml file.
        yml_filename (str): Name of the yml file.
        record_id (str): Record id.
        add (bool, optional): True to add a record to the list. Defaults to False.
        remove (bool, optional): True to remove a record from the list. Defaults to False.
    """
    os.chdir(yml_dir)
    info=_load_yaml(yml_filename)
        
    excluded=info['excluded_records']
    if add and not remove:
        for record in record_id:
            excluded.append(record)
    elif not add and remove:
        for record in record_id:
            excluded.remove(record)
    elif add and remove:
        raise ValueError("Both 'add' and 'remove' cannot be True.")
    else:
        raise ValueError("Either 'add' or 'remove' must be True.") 
        
    info['excluded_records']=excluded
    
    _save_yaml(yml_filename,info)


def _load_yaml(yml_filename):

    with open(yml_filename, "r") as recordfile:
        info=yaml.safe_load(recordfile)

    return info


def _save_yaml(yml_filename,info):

    with open(yml_filename,"w") as recordfile:
        yaml.dump(info, recordfile) 
