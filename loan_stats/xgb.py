import os 
from datetime import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import cross_validation, metrics   
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc 
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import seaborn as sns
#%matplotlib inline

wd = '/Users/ewenwang/OneDrive/IBM/Project_ICBC/code_wu'
os.chdir(wd)

def loaddata(wd, DataFile, seed = 7, test_size = 0.33):
    """ Load data and split it into training and test datasets """
    
    dataset = pd.read_csv(DataFile)
    target = 'loan_status'
    predictors = [x for x in dataset.columns if x not in [target]]
    dtrain, dtest = train_test_split(dataset, test_size=test_size, random_state=seed)
    
    return dtrain, dtest

def modelfit(alg, dtrain, dtest, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    """ Fit models w/ parameters """
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgtest = xgb.DMatrix(dtest[predictors].values)
        cvresult = xgb.cv(xgb_param, 
                          xgtrain, 
                          num_boost_round=alg.get_params()['n_estimators'], 
                          nfold=cv_folds,
                          metrics='auc', 
                          early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
        print(cvresult)
    
    alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')

    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    dtest['predprob'] = alg.predict_proba(dtest[predictors])[:,1]
        
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob)) 
    print('AUC Score (Test): %f' % metrics.roc_auc_score(dtest[target], dtest['predprob']))
    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

def xgbPT():
    """ Store Tuned Parameters """
    
    xgbFinal = XGBClassifier(
        learning_rate=0.01,
        n_estimators=5000,
        max_depth=3,
        min_child_weight=2,
        gamma=0,
        subsample=0.4,
        colsample_bytree=0.7,
        reg_alpha=1,
        objective= 'binary:logistic',
        scale_pos_weight=1,
        seed=2017)
    return xgbFinal

def main():
	""" Combine functions """
	
	#DataFile = 'weizhu.csv'
	rcParams['figure.figsize'] = 12, 4
	DataFile = "dataset_fe.csv"
    dtrain, dtest = loaddata(wd, DataFile)
    xgbFinal = xgbPT()
    datetime.datetime.now()
    modelfit(xgbFinal, dtrain, dtest, predictors)
    print('Time Cost: ', datetime.datetime.now() - st)

if __name__ == "__main__":
	main()
