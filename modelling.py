import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from scipy import stats

class modelling():
    def __init__(self):
        print ("Welcome to modelling part of the pacakage")

    def open_svm(self, df_train, response_var, dict_paramters = {}, performCV = 1, cv_folds = 10, printKS = 1, printAUCcurve = 1, random_state = 29):
        X = df_train.copy()

        X_train = X.loc[:, X.columns != response_var]
        col_names = X.columns

        y_train = np.array(X.loc[:,response_var])

        if 'C' in dict_paramters:
            c = dict_paramters['C']
        else:
            c = 0.01
        
        if 'kernel' in dict_paramters:
            kernel = dict_paramters['kernel']
        else:
            kernel = 'rbf'

        if 'degree' in dict_paramters:
            degree = dict_paramters['degree']
        else:
            degree = 3
        
        if 'random_State' in dict_paramters:
            random_state = dict_paramters['random_state']
        else:
            random_state = 29

        if 'max_iter' in dict_paramters:
            max_iter = dict_paramters['max_iter']
        else:
            max_iter = 100

        if 'probability' in dict_paramters:
            probability = dict_paramters['probability']
        else:
            probability = False

        clf_open_svm = SVC(C= c, kernel = kernel, degree = degree, probability = probability, max_iter = max_iter, random_state = random_state)
        clf_open_svm.fit(X_train, y_train) 
        
        # prob > 0.5 => 1 else 0
        hard_predictions_train = clf_open_svm.predict(X_train)
        
        # considering only class = 1: either binary or one-vs-all
        soft_predictions_train = clf_open_svm.predict_proba(X_train)[:,1]

        if performCV:
            cv_score = cross_val_score(clf_open_svm, X_train, y_train, cv = cv_folds, scoring = 'roc_auc')
        
        print ("\n###########################################")
        print ("\n#############TRAINING RESULTS##############")
        print ("\n###########################################")
        print ("\nModel Report")

        print ("AUC Score (Train): %f" % roc_auc_score(y_train, soft_predictions_train))
        
        if performCV:
            print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))

        if printKS:
            print ("\n#### KS and p-val on Train set####")
            metric_ks(soft = soft_predictions_train, target = y_train)
        
        if printAUCcurve:
            print ("\n#### ROC curve (Train set)####")
            metric_auc(soft = soft_predictions_train, target = y_train)
            
        return clf_open_svm, soft_predictions_train, hard_predictions_train

    def open_naive_bayes(self, df_train, response_var, dict_paramters = {}, performCV = 1, cv_folds = 10, printKS = 1, printAUCcurve = 1):
        
        X = df_train.copy()
        X_train = X.loc[:, X.columns != response_var]
        y_train = np.array(X.loc[:, response_var])
        
        if 'priors' in dict_paramters:
            priors = dict_paramters['priors']
        else:
            priors=None, 
            
        if 'var_smoothing' in dict_paramters:
            var_smoothing = dict_paramters['var_smoothing']
        else:
            var_smoothing = 1e-09

        clf_open_nb = GaussianNB(priors = priors, var_smoothing = var_smoothing)
        clf_open_nb.fit(X_train, y_train)

        # prob > 0.5 => 1 else 0
        hard_predictions_train = clf_open_nb.predict(X_train)
        
        # considering only class = 1: either binary or one-vs-all
        soft_predictions_train = clf_open_nb.predict_proba(X_train)[:,1]

        if performCV:
            cv_score = cross_val_score(clf_open_nb, X_train, y_train, cv = cv_folds, scoring = 'roc_auc')
        
        print ("\n###########################################")
        print ("\n#############TRAINING RESULTS##############")
        print ("\n###########################################")
        print ("\nModel Report")

        print ("AUC Score (Train): %f" % roc_auc_score(y_train, soft_predictions_train))
        
        if performCV:
            print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))

        if printKS:
            print ("\n#### KS and p-val on Train set####")
            metric_ks(soft = soft_predictions_train, target = y_train)
        
        if printAUCcurve:
            print ("\n#### ROC curve (Train set)####")
            metric_auc(soft = soft_predictions_train, target = y_train)
            
        return clf_open_nb, soft_predictions_train, hard_predictions_train   