import os
import numpy as np
import pandas as pd

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
from sklearn.inspection import permutation_importance
import optuna

def RF_based_feature_selection(X,y,cv,n_trials=1000):
    
    n_splis = cv.get_n_splits(X)
    
    feature_importance = np.zeros((n_splis,X.shape[-1]))
    
    scaler = StandardScaler
    classifier = RandomForestClassifier
    
    for i,(train_index,test_index) in enumerate(cv.split(X,y)):
        train_X, train_y, test_X, test_y = X[train_index],y[train_index],X[test_index],y[test_index]
        
        def objective(trial):
            trial.set_user_attr("random_state",42)
            params = {
                "n_estimators":trial.suggest_int("n_estimators", 1, 1e+3),
                "criterion":trial.suggest_categorical("criterion", ['gini', 'entropy', 'log_loss']),
                "max_depth":trial.suggest_int("max_depth", 5, 1e+3),
                "max_features":trial.suggest_categorical("max_features", ['sqrt', 'log2']),
                "bootstrap":trial.suggest_categorical("bootstrap", [True, False]),
                "random_state":trial.user_attrs["random_state"],
            }

            # define the pipeline
            steps = list()
            steps.append(('scaler', scaler()))
            steps.append(('model', classifier(**params)))
            pipeline = Pipeline(steps=steps)

            pipeline.fit(train_X,train_y)
            pred = pipeline.predict(test_X)
            score = accuracy_score(test_y,pred)
            return score
        
        study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(n_startup_trials=int(n_trials*0.1)))
        study.optimize(objective,n_trials=n_trials)
        
        # define the pipeline
        steps = list()
        steps.append(('scaler', scaler()))
        steps.append(('model', classifier(**study.best_params,**study.best_trial.user_attrs)))
        pipeline = Pipeline(steps=steps)
        
        pipeline.fit(train_X,train_y)
        feature_importance[i] = pipeline["model"].feature_importances_
        
    return feature_importance
        
        
        