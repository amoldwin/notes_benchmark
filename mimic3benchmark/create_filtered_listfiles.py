import pandas as pd
import os

def is_kept(ts_filename,t):
    ts = pd.read_csv(ts_filename)
    if not t==None:
        time_filter = ts['Hours'].apply(lambda x: x<=(t+1e-6))
        ts=ts[time_filter]
    cuis_empty = ts.CUIS.isnull().all()
    words_empty = ts.WORDS.isnull().all()
    d2v_empty = ts.DOC2VEC.isnull().all()
    non_structured = ['CUIS','WORDS','DOC2VEC']
    
    structured_empty = False
    for col in [col for col in ts.columns if not col in non_structured]:
        structured_empty = structured_empty and ts[col].isnull().all()

    keep = not (cuis_empty or words_empty or d2v_empty or structured_empty)
    return keep

for directory_path in ['./data/in-hospital-mortality', './data/decompensation','./data/phenotyping','./data/length-of-stay','./data/multitask']:
    sets = ['train','val','test']
    set_paths =[os.path.join(directory_path, x) for x in ['train/','train/','test/']]


    for i,set in enumerate(sets):
        filename = set+'_listfile.csv'
#          print(filename)
        df = pd.read_csv(os.path.join(directory_path, filename))
        set_path = set_paths[i]
        
        if 'period_length' not in df.columns:
            keep = df.apply(lambda x: is_kept(os.path.join(set_path, x.stay),None),axis=1)
        else:
            keep = df.apply(lambda x: is_kept(os.path.join(set_path, x.stay),x.period_length),axis=1)
    

        df[keep].to_csv(os.path.join(directory_path, 'filtered_'+filename), index=False)