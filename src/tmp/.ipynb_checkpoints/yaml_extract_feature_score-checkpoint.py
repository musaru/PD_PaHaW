import os
import yaml
import numpy as np
import pandas as pd

def main():
    with open("../outputs/all_in_one.yaml","r") as f:
        data = yaml.load(f,Loader=yaml.Loader)
        
    # print(data)
    
    feature_name_and_score = {key:data[key]["score"] for key in data.keys()}
    
    score_sorted = sorted(feature_name_and_score.items(), key=lambda x:x[1],reverse=True)
    df = pd.DataFrame({key:value for key,value in score_sorted},index=['accuracy'])
    df.to_csv("task_feature_wise_performance_on_loo.csv")
    
if __name__=="__main__":
    main()