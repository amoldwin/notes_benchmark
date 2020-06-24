import pandas as pd
import os
def is_kept(ts_filename):
    ts = pd.read_csv(ts_filename)
    cuis_empty = ts.CUIS.isnull().all()
    words_empty = ts.WORDS.isnull().all()
    d2v_empty = ts.DOC2VEC.isnull().all()
    non_structured = ['CUIS','WORDS','DOC2VEC']
    
    structured_empty = False
    for col in [col for col in ts.columns if not col in non_structured]:
        structured_empty = structured_empty and ts[col].isnull().all()

    keep = not (cuis_empty or words_empty or d2v_empty or structured_empty)
    return keep
for directory_path in ['../data/in-hospital-mortality', '../data/decompensation','../data/phenotyping','../data/length-of-stay','../data/multitask']:
    sets = ['train','val','test']
    set_paths =[os.path.join(directory_path, x) for x in ['train/','train/','test/']]


    for i,set in enumerate(sets):
        filename = set+'_listfile.csv'
#          print(filename)
        df = pd.read_csv(os.path.join(directory_path, filename))
        set_path = set_paths[i]
        keep = df.stay.apply(lambda x: is_kept(os.path.join(set_path, x)))
        df[keep].to_csv(os.path.join(directory_path, 'filtered_'+filename), index=False)