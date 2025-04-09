import os
import numpy as np
import pandas as pd

from tqdm import tqdm

from tsfresh import extract_features

def load_HW_data():
    subject_level_hw_data_paths = dict()
    for subject_id in os.listdir("../../PaHaW/PaHaW_public"):
        for file in tqdm(os.listdir(os.path.join("../../PaHaW/PaHaW_public",subject_id))):
            if file.endswith("svc"):
                if int(subject_id) not in subject_level_hw_data_paths.keys():
                    subject_level_hw_data_paths[int(subject_id)] = [os.path.join("../../PaHaW/PaHaW_public",subject_id,file)]
                else:
                    subject_level_hw_data_paths[int(subject_id)] += [os.path.join("../../PaHaW/PaHaW_public",subject_id,file)]
    return subject_level_hw_data_paths
                
    

def read_label_info():
    info = pd.read_excel('../../PaHaW/PaHaW_files/corpus_PaHaW.xlsx')
    ID_Disease = dict(info[['ID','Disease']].values)
    print(ID_Disease)
    
def main():
    subject_level_hw_data_paths = load_HW_data()
    for subject_id in subject_level_hw_data_paths.keys():
        for file_path in subject_level_hw_data_paths[subject_id]:
            df = pd.read_csv(file_path, header=None,names=["Y", "X", "time","button","azimuth", "altitude", "pressure"],sep=" ",skiprows=1)
                        
            
            df["id"] = f"{subject_id}__{os.path.splitext(os.path.basename(file_path))[0][-3:]}"
            
            X = extract_features(df, column_id='id', column_sort='time',n_jobs=8)
            # print(set([base.split("__")[0] for base in X.columns]))
            print(X.shape)
            break
        break
    
    
if __name__=="__main__":
    main()