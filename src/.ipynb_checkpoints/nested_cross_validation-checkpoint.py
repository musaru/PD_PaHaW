import os
import sys
sys.setrecursionlimit(int(1e+6))

import numpy as np
import pandas as pd
import scipy
import yaml

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


def nested_cross_validation(X,y,cv=StratifiedKFold(n_splits=10),output_path="./outputs",output_file_name="output"):
    
    classification_reports = dict()
    
    scaler = StandardScaler
    classifier_func_with_optuna = classification_by_svc
    
    for n_fold, (train_set_index,test_set_index) in enumerate(cv.split(X,y)):
        train_X,train_y,test_X,test_y = X[train_set_index],y[train_set_index],X[test_set_index],y[test_set_index]
        
        _, best_param, classifier =classifier_func_with_optuna(train_X,train_y,n_trials=int(1e+3),cv=cv)
        
        steps = list()
        steps.append(('scaler', scaler()))
        steps.append(('model', classifier(**best_param)))
        pipeline = Pipeline(steps=steps)
        
        pipeline.fit(train_X,train_y)
        pred = pipeline.predict(test_X)
        
        classification_reports[f"{n_fold+1:02}_fold"] = classification_report(test_y,pred,digits=4,output_dict=True)
        
        with open(os.path.join(output_path,f'{output_file_name}.yaml'),'w')as f:
            yaml.dump(classification_reports, f, default_flow_style=False, allow_unicode=True)
    
    cv_accuracy = []
    for key in classification_reports.keys():
        
        cv_accuracy.append(classification_reports[key]["accuracy"])
        
    cv_accu = np.mean(cv_accuracy)
        
    classification_reports["avg_accuracy"] = float(np.mean(cv_accuracy))
    classification_reports["var_accuracy"] = float(np.var(cv_accuracy))
    classification_reports["max_accuracy"] = float(np.max(cv_accuracy))
    classification_reports["min_accuracy"] = float(np.min(cv_accuracy))
    
    with open(os.path.join(output_path,'nested_cv_result.yaml'),'w')as f:
            yaml.dump(classification_reports, f, default_flow_style=False, allow_unicode=True)
            
    return classification_reports
                                                               