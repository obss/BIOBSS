import os
import yaml


def add_record(yml_dir,yml_filename,record_id,file_types,marker,events):
    
    os.chdir(yml_dir)
    
    with open(yml_filename, "r") as recordfile:
        info=yaml.safe_load(recordfile)
        
    records=info['records']
    records[record_id]={'file_types':file_types,'marker':marker,'events':events}
    info['records']=records
    
    with open(yml_filename,"w") as recordfile:
    
        yaml.dump(info, recordfile)


def update_excluded(yml_dir,yml_filename,record_id,add=False,remove=False):
    
    os.chdir(yml_dir)
    
    with open(yml_filename, "r") as recordfile:
        info=yaml.safe_load(recordfile)
        
    excluded=info['excluded_records']
    if add:
        for record in record_id:
            excluded.append(record)
    if remove:
        for record in record_id:
            excluded.remove(record)
        
    info['excluded_records']=excluded
    
    with open(yml_filename,"w") as recordfile:
    
        yaml.dump(info, recordfile)


def set_parameters(yml_dir,yml_filename,record_id,parameters):
    
    os.chdir(yml_dir)
    
    with open(yml_filename, "r") as recordfile:
        info=yaml.safe_load(recordfile)
    
    par=info['parameters']
    par[record_id]=parameters
    info['parameters']=par
    
    with open(yml_filename,"w") as recordfile:
    
        yaml.dump(info, recordfile)        
    