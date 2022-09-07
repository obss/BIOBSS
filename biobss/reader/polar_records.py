import os
import yaml

def add_record(yml_dir: str, yml_filename: str, record_id: str, file_types: list, marker: bool, events: list, bp: bool, fs_hr: int=None, fs_ppg: int=None):
    """Adds a new record to a yml file.

    Args:
        yml_dir (str): Directory of the yml file.
        yml_filename (str): Name of the yml file.
        record_id (str): Record id.
        file_types (list): File types included in the record.
        marker (bool): True if marker file exists.
        events (list): Event list of the record.
        bp (bool): True if blood pressure measurement exists.
        fs_ppg (int, optional): Sampling rate of the PPG signal. Defaults to None.
        fs_hr (int, optional): Sampling rate of the HR signal. Defaults to None.
    """
    os.chdir(yml_dir)   
    info=_load_yaml(yml_filename)
        
    records=info['records']
    records[record_id]={'file_types':file_types,'marker':marker,'events':events,'bp':bp,'fs_ppg':fs_ppg,'fs_hr':fs_hr}
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
    if add:
        for record in record_id:
            excluded.append(record)
    if remove:
        for record in record_id:
            excluded.remove(record)
        
    info['excluded_records']=excluded
    
    _save_yaml(yml_filename,info)


def _load_yaml(yml_filename):

    with open(yml_filename, "r") as recordfile:
        info=yaml.safe_load(recordfile)

    return info


def _save_yaml(yml_filename,info):

    with open(yml_filename,"w") as recordfile:
        yaml.dump(info, recordfile) 
