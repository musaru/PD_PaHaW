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
from sklearn.tree import (
    DecisionTreeClassifier
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

# optuna.logging.disable_default_handler()

# SVC
def classification_by_svc(X:np.ndarray,y:np.ndarray,n_trials=1e+3,cv = StratifiedKFold(10,shuffle=True,random_state=42)):
    
    scaler = StandardScaler
    classifier = SVC
    
    def objective(trial):
        trial.set_user_attr("random_state",42)
        # trial.set_user_attr("max_iter",int(1e+8))
        params = {
            "C":trial.suggest_float("C", 1e-3, 1e+3),
            "kernel":trial.suggest_categorical("kernel", [
                # 'linear', 
                # 'poly', 
                'rbf',
                'sigmoid'
            ]),
            "gamma":trial.suggest_float("gamma",1e-3, 1e+3),
            "degree":trial.suggest_int("degree",1,5),
            "coef0":trial.suggest_float("coef0",1e-3, 1e+3),
            # "max_iter":trial.user_attrs["max_iter"],
            "random_state":trial.user_attrs["random_state"],
        }

        # define the pipeline
        steps = list()
        steps.append(('scaler', scaler()))
        steps.append(('model', classifier(**params)))
        pipeline = Pipeline(steps=steps)

        # evaluate the model using cross-validation
        scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        return scores.mean()
                         
    study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(n_startup_trials=int(n_trials*0.1)))
    study.optimize(objective,n_trials=n_trials)
                         

    # define the pipeline
    steps = list()
    steps.append(('scaler', scaler()))
    steps.append(('model', classifier(**study.best_params,**study.best_trial.user_attrs)))
    pipeline = Pipeline(steps=steps)
    # evaluate the model using cross-validation
    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print(f"Accuracy : {scores.mean()}\tparam : {study.best_params,study.best_trial.user_attrs}")
    
    return scores.mean(),study.best_params|study.best_trial.user_attrs, classifier

# ExtraTreesClassifier (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier)
def classification_by_et(X:np.ndarray,y:np.ndarray,n_trials=1e+3,cv = StratifiedKFold(10,shuffle=True,random_state=42)):

    
    scaler = StandardScaler
    classifier = ExtraTreesClassifier
    
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

        # evaluate the model using cross-validation
        scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        return scores.mean()
                         
    study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(n_startup_trials=int(n_trials*0.1)))
    study.optimize(objective,n_trials=n_trials)
                         

    # define the pipeline
    steps = list()
    steps.append(('scaler', scaler()))
    steps.append(('model', classifier(**study.best_params,**study.best_trial.user_attrs)))
    pipeline = Pipeline(steps=steps)
    # evaluate the model using cross-validation
    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print(f"Accuracy : {scores.mean()}\tparam : {study.best_params,study.best_trial.user_attrs}")
    
    return scores.mean(),study.best_params|study.best_trial.user_attrs, classifier

# RandomForestClassifier (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)
def classification_by_rf(X:np.ndarray,y:np.ndarray,n_trials=1e+3,cv=StratifiedKFold(10,shuffle=True,random_state=42)):
    
    
    scaler = StandardScaler
    classifier = RandomForestClassifier
    
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

        # evaluate the model using cross-validation
        scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        return scores.mean()
                         
    study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(n_startup_trials=int(n_trials*0.1)))
    study.optimize(objective,n_trials=n_trials)
                         

    # define the pipeline
    steps = list()
    steps.append(('scaler', scaler()))
    steps.append(('model', classifier(**study.best_params,**study.best_trial.user_attrs)))
    pipeline = Pipeline(steps=steps)
    # evaluate the model using cross-validation
    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print(f"Accuracy : {scores.mean()}\tparam : {study.best_params,study.best_trial.user_attrs}")
    
    return scores.mean(),study.best_params|study.best_trial.user_attrs, classifier


# GradientBoostingClassifier (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier)
def classification_by_gb(X:np.ndarray,y:np.ndarray,n_trials=1e+3,cv=StratifiedKFold(10,shuffle=True,random_state=42)):

    
    scaler = StandardScaler
    classifier = GradientBoostingClassifier
    
    def objective(trial):
        trial.set_user_attr("random_state",42)
        params = {
            "loss":trial.suggest_categorical("loss", ['log_loss', 'exponential']),
            "learning_rate":trial.suggest_float("learning_rate",1e-6,1),
            "n_estimators":trial.suggest_int("n_estimators", 1, 1e+3),
            "subsample":trial.suggest_float("subsample",1e-6,1),
            "criterion":trial.suggest_categorical("criterion",["friedman_mse", "squared_error"]),
            "max_depth":trial.suggest_int("max_depth", 5, 1e+3),
            "max_features":trial.suggest_categorical("max_features", ['sqrt', 'log2']),
            "warm_start":trial.suggest_categorical("warm_start",[True,False]),
            "random_state":trial.user_attrs["random_state"],
        }

        # define the pipeline
        steps = list()
        steps.append(('scaler', scaler()))
        steps.append(('model', classifier(**params)))
        pipeline = Pipeline(steps=steps)

        # evaluate the model using cross-validation
        scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        return scores.mean()
                         
    study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(n_startup_trials=int(n_trials*0.1)))
    study.optimize(objective,n_trials=n_trials)
                         

    # define the pipeline
    steps = list()
    steps.append(('scaler', scaler()))
    steps.append(('model', classifier(**study.best_params,**study.best_trial.user_attrs)))
    pipeline = Pipeline(steps=steps)
    # evaluate the model using cross-validation
    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print(f"Accuracy : {scores.mean()}\tparam : {study.best_params,study.best_trial.user_attrs}")
    
    return scores.mean(),study.best_params|study.best_trial.user_attrs, classifier

# AdaBoostClassifier (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier)
def classification_by_ab(X:np.ndarray,y:np.ndarray,n_trials=1e+3,cv=StratifiedKFold(10,shuffle=True,random_state=42)):

    
    scaler = StandardScaler
    classifier = AdaBoostClassifier
    
    def objective(trial):
        trial.set_user_attr("estimator",DecisionTreeClassifier(max_depth=1,random_state=42))
        trial.set_user_attr("random_state",42)
        trial.set_user_attr("algorithm",'SAMME')
        params = {
            "estimator":trial.user_attrs["estimator"],
            "n_estimators":trial.suggest_int("n_estimators", 1, 1e+3),
            "learning_rate":trial.suggest_float("learning_rate",1e-6,1),
            "algorithm":trial.user_attrs["algorithm"],
            "random_state":trial.user_attrs["random_state"],
        }

        # define the pipeline
        steps = list()
        steps.append(('scaler', scaler()))
        steps.append(('model', classifier(**params)))
        pipeline = Pipeline(steps=steps)

        # evaluate the model using cross-validation
        scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        return scores.mean()
                         
    study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(n_startup_trials=int(n_trials*0.1)))
    study.optimize(objective,n_trials=n_trials)
                         

    # define the pipeline
    steps = list()
    steps.append(('scaler', scaler()))
    steps.append(('model', classifier(**study.best_params,**study.best_trial.user_attrs)))
    pipeline = Pipeline(steps=steps)
    # evaluate the model using cross-validation
    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print(f"Accuracy : {scores.mean()}\tparam : {study.best_params,study.best_trial.user_attrs}")
    
    return scores.mean(),study.best_params|study.best_trial.user_attrs, classifier

# KNeighborsClassifier (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)
def classification_by_knn(X:np.ndarray,y:np.ndarray,n_trials=1e+3,cv=StratifiedKFold(10,shuffle=True,random_state=42)):
    scaler = StandardScaler
    classifier = KNeighborsClassifier
    
    def objective(trial):
        params = {
            "n_neighbors":trial.suggest_int("n_neighbors", 3, 29, step=2),
            "weights":trial.suggest_categorical("weights",["uniform","distance"]),
            "algorithm":trial.suggest_categorical("algorithm",["auto","ball_tree","kd_tree","brute"]),
        }

        # define the pipeline
        steps = list()
        steps.append(('scaler', scaler()))
        steps.append(('model', classifier(**params)))
        pipeline = Pipeline(steps=steps)

        # evaluate the model using cross-validation
        scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        return scores.mean()
                         
    study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(n_startup_trials=int(n_trials*0.1)))
    study.optimize(objective,n_trials=n_trials)
                         

    # define the pipeline
    steps = list()
    steps.append(('scaler', scaler()))
    steps.append(('model', classifier(**study.best_params,**study.best_trial.user_attrs)))
    pipeline = Pipeline(steps=steps)
    # evaluate the model using cross-validation
    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print(f"Accuracy : {scores.mean()}\tparam : {study.best_params,study.best_trial.user_attrs}")
    
    return scores.mean(),study.best_params|study.best_trial.user_attrs, classifier