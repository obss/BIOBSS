import os
import yaml


def add_record(yml_dir,yml_filename,record_id,file_types,marker,events):
    
    os.chdir(yml_dir)   
    info=_load_yaml(yml_filename)
        
    records=info['records']
    records[record_id]={'file_types':file_types,'marker':marker,'events':events}
    info['records']=records
 
    _save_yaml(yml_filename,info)


def update_excluded(yml_dir,yml_filename,record_id,add=False,remove=False):
    
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


def set_parameters(yml_dir,yml_filename,record_id,parameters):
    
    os.chdir(yml_dir)
    
    info=_load_yaml(yml_filename)
    
    par=info['parameters']
    par[record_id]=parameters
    info['parameters']=par
    
    _save_yaml(yml_filename,info)        
    

def _load_yaml(yml_filename):

    with open(yml_filename, "r") as recordfile:
        info=yaml.safe_load(recordfile)

    return info


def _save_yaml(yml_filename,info):

    with open(yml_filename,"w") as recordfile:
        yaml.dump(info, recordfile) 
