import os
import sys
sys.setrecursionlimit(int(1e+6))

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
)

from sklearn.svm import SVC
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    
)
from sklearn.neighbors import (
    KNeighborsClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    r2_score,
    recall_score
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
    train_test_split,
    LeaveOneOut
)

from sklearn.pipeline import Pipeline
import optuna

from ml_classification import (
    classification_by_svc, # SVC
    classification_by_et, # ExtraTreesClassifier
    classification_by_rf, # RandomForestClassifier
    classification_by_gb, # GradientBoostingClassifier
    classification_by_ab, # AdaBoostClassifier
    classification_by_knn, # KNeighborsClassifier
)
from feature_selection import (
    RF_based_feature_selection,
)

from nested_cross_validation import nested_cross_validation
from collections import defaultdict

def main():
    random_state = 42
    csv_path = "../PaHaW_extracted_files/1_1.csv"
    df = pd.read_csv(csv_path,index_col=0).reset_index(drop=True)
    feature_names = df.columns[:-2] 
    label = df.columns[-2]
    X,y = df[feature_names].to_numpy(),df[label].to_numpy()
    
    
    selected_index = None
    cv=StratifiedKFold(10,shuffle=True,random_state=random_state)
    # cv=LeaveOneOut()
    
    feature_importance = RF_based_feature_selection(X,y,cv=cv,n_trials=100)
    feature_ranking = np.argsort(np.sum(feature_importance,axis=0))[::-1]
    
    selected_index = feature_ranking[:-int(X.shape[-1]*0.5)]
    print(selected_index)
    
    # scaler = StandardScaler
    _, best_param, classifier =classification_by_rf(
        # X,
        X[:,selected_index],
        y,
        n_trials=int(1e+3),
        cv=cv
    )
    
def nested_cv_main():
    output_path = "./outputs"
    
    os.makedirs(output_path,exist_ok=True)
    
    random_state = 42
    csv_path = "../PaHaW_extracted_files/1_1.csv"
    df = pd.read_csv(csv_path,index_col=0).reset_index(drop=True)
    feature_names = df.columns[:-2] 
    label = df.columns[-2]
    X,y = df[feature_names].to_numpy(),df[label].to_numpy()
    
    cv =StratifiedKFold(n_splits=10,shuffle=True,random_state=random_state)
    cv = LeaveOneOut()
    
    nested_cross_validation(X,y,cv=cv,output_path=output_path)
    
    
def feature_wise_nested_cv_main():
    output_path = "./outputs"
    
    os.makedirs(output_path,exist_ok=True)
    
    random_state = 42
    csv_path = "../PaHaW_extracted_files/1_1.csv"
    df = pd.read_csv(csv_path,index_col=0).reset_index(drop=True)
    feature_names = df.columns[:-2] 
    label = df.columns[-2]
    
    results = dict()
    
    for feature in tqdm(feature_names):
        X,y = df[feature].to_numpy().reshape(-1,1),df[label].to_numpy()

        cv =StratifiedKFold(n_splits=10,shuffle=True,random_state=random_state)
        cv = LeaveOneOut()

        classification_reports = nested_cross_validation(X,y,cv=cv,output_path=output_path,output_file_name=feature)
        
        results[feature]=classification_reports["avg_accuracy"]
        with open(os.path.join(output_path,'all_in_one.yaml'),'w')as f:
            yaml.dump(results, f, default_flow_style=False, allow_unicode=True)


def feature_wize_cv_main():
    
    _df = defaultdict(list)
    
    output_path = "./outputs"
    
    os.makedirs(output_path,exist_ok=True)
    
    random_state = 42
    csv_path = "../PaHaW_extracted_files/4_1.csv"
    df = pd.read_csv(csv_path,index_col=0).reset_index(drop=True)
    feature_names = df.columns[:-2] 
    label = df.columns[-2]
    
    results = dict()
    
    for feature in tqdm(feature_names):
        X,y = df[feature].to_numpy().reshape(-1,1),df[label].to_numpy()

        # cv =StratifiedKFold(n_splits=10,shuffle=True,random_state=random_state)
        cv = LeaveOneOut()

        score,param,clf = classification_by_svc(X,y,cv=cv)
        
        _df["feature"].append(feature)
        _df["score"].append(score)
        _df["param"].append(param)
        # results[feature]={"score":float(score),
        #                   "param":param,
        #                  }
        # with open(os.path.join(output_path,'all_in_one.yaml'),'w')as f:
        #     yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
        pd.DataFrame(_df).to_csv(os.path.join(output_path,'featurewise_loo_result_task4.csv'))

    
    
    
    
if __name__=="__main__":
    feature_wize_cv_main()