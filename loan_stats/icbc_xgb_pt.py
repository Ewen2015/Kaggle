##############################################################################
# 1: Package Importing
##############################################################################

print('Importing packages...')
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
rcParams['figure.figsize'] = 12, 4


######################################################################################
# 2: Data Loading
######################################################################################

print('Setting the work director...')
wd = '/Users/ewenwang/OneDrive/IBM/Project_ICBC/code_wu'
os.chdir(wd)

print('Loading the data...')
DataFile = "dataset_new.csv"
dataset = pd.read_csv(DataFile)
target = 'loan_status'
predictors = [x for x in dataset.columns if x not in [target]]

# split data into train and test sets
seed, test_size = 2017, .33
dtrain, dtest = train_test_split(dataset, test_size=test_size, random_state=seed)

print('Done.')

######################################################################################
# 3: Parameter Tuning
######################################################################################

# 1. handle imbalanced datasets:
    
# 1) care only about the ranking order (AUC): 
# - use AUC for evaluation
# - balance the positive and negative weights, via scale_pos_weight
# 2) care about predicting the right probability:
# - cannot re-balance the dataset
# - set parameter max_delta_step to a finite number (say 1) will help convergence

# 2. control overfitting:

# 1) control model complexity: max_depth, min_child_weight and gamma
# 2) add randomness to make training robust to noise: subsample, colsample_bytree; 
# can also reduce stepsize eta, but needs to remember to increase num_round.


### Parameters Tuning 
target = 'loan_status'
predictors = [x for x in dataset.columns if x not in [target]]


def modelfit(alg, dtrain, dtest, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        print('Start Cross Validation...')
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
        print('Done.')
    
    #Fit the algorithm on the data
    print('Fitting the algorithm on the data...')
    alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')
        
    #Predict training set:
    print('Predicting training set...')
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob)) 
    
    #Predict on testing data:
    print('Predicting testing set...')
    dtest['predprob'] = alg.predict_proba(dtest[predictors])[:,1]
    print('AUC Score (Test): %f' % metrics.roc_auc_score(dtest[target], dtest['predprob']))
    
    #Print featuure importance:   
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()

## Step 1: Fix learning rate and number of estimators for tuning tree-based parameters
xgb1 = XGBClassifier(learning_rate =0.1,
                     n_estimators=1000,
                     max_depth=5,
                     min_child_weight=1,
                     gamma=0,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     scale_pos_weight=1,
                     objective= 'binary:logistic',
                     seed=2017)
modelfit(xgb1, dtrain, dtest, predictors)

lr, ns = 0.1, 65
# AUC Score (Test): 0.699657

## Step 2: Tune max_depth and min_child_weight
param_test1 = {
    'max_depth': list(range(1,5,1)),
    'min_child_weight': list(range(1,6,1))
}

rsearch1 = RandomizedSearchCV(estimator = XGBClassifier(
    learning_rate=lr, 
    n_estimators=ns, 
    max_depth=5,
    min_child_weight=1, 
    gamma=0, 
    subsample=0.8, 
    colsample_bytree=0.8,
    objective= 'binary:logistic', 
    scale_pos_weight=1, 
    seed=2017), 
    param_distributions = param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=5)

rsearch1.fit(dtrain[predictors], dtrain[target])
rsearch1.best_params_, rsearch1.best_score_

md, mcw = 
# 

param_test2 = {
 'max_depth': [2, 3, 4],
 'min_child_weight': [1,2,3]
}

gsearch2 = GridSearchCV(estimator = XGBClassifier(
    learning_rate=lr, 
    n_estimators=ne, 
    max_depth=5,
    min_child_weight=1, 
    gamma=0, 
    subsample=0.8, 
    colsample_bytree=0.8,
    objective= 'binary:logistic', 
    scale_pos_weight=1, 
    seed=2017), 
    param_grid = param_test2, scoring='roc_auc', n_jobs=4, iid=False, cv=5)

gsearch2.fit(dtrain[predictors], dtrain[target])
gsearch2.best_params_, gsearch2.best_score_

#({'max_depth': 3, 'min_child_weight': 2}, 0.6933633546997919)

## Step 3: Tune gamma
param_test3 = {
    'gamma':[i/10.0 for i in range(0,5)]
}

gsearch3 = GridSearchCV(estimator = XGBClassifier(
    learning_rate=lr, 
    n_estimators=ne, 
    max_depth=md,
    min_child_weight=mcw, 
    gamma=0, 
    subsample=0.8, 
    colsample_bytree=0.8,
    objective= 'binary:logistic', 
    scale_pos_weight=1, 
    seed=2017), 
    param_grid = param_test3, scoring='roc_auc', n_jobs=4, iid=False, cv=5)

gsearch3.fit(dtrain[predictors], dtrain[target])
gsearch3.best_params_, gsearch3.best_score_

g = 
#({'gamma': 0.0}, 0.6933633546997919)

xgb2 = XGBClassifier(
    learning_rate=lr,
    n_estimators=1000,
    max_depth=md,
    min_child_weight=mcw,
    gamma=g,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    scale_pos_weight=1,
    seed=2017)
modelfit(xgb2, dtrain, dtest, predictors)

#68

## Step 4: Tune subsample and colsample_bytree
param_test4 = {
    'subsample':[i/10.0 for i in range(4,8)],
    'colsample_bytree':[i/10.0 for i in range(6,10)]
}

rsearch4 = RandomizedSearchCV(estimator = XGBClassifier(
    learning_rate=lr, 
    n_estimators=ne, 
    max_depth=md,
    min_child_weight=mcw, 
    gamma=g, 
    subsample=0.8, 
    colsample_bytree=0.8,
    objective= 'binary:logistic', 
    scale_pos_weight=1, 
    seed=2017), 
    param_distributions = param_test4, scoring='roc_auc', n_jobs=4, iid=False, cv=5)

rsearch4.fit(dtrain[predictors], dtrain[target])
rsearch4.best_params_, rsearch4.best_score_

c, cb = 
#({'colsample_bytree': 0.7, 'subsample': 0.5}, 0.69626054920530689)

param_test5 = {
    'subsample':[i/100.0 for i in range(40,60,5)],
    'colsample_bytree':[i/100.0 for i in range(60,80,5)]
}

rsearch5 = RandomizedSearchCV(estimator = XGBClassifier(
    learning_rate=lr, 
    n_estimators=ne, 
    max_depth=md,
    min_child_weight=mcw, 
    gamma=g, 
    subsample=s, 
    colsample_bytree=cb,
    objective= 'binary:logistic', 
    scale_pos_weight=1, 
    seed=2017), 
    param_distributions = param_test5, scoring='roc_auc', n_jobs=4, iid=False, cv=5)

rsearch5.fit(dtrain[predictors], dtrain[target])
rsearch5.best_params_, rsearch5.best_score_

#({'colsample_bytree': 0.7, 'subsample': 0.5}, 0.69626054920530689)

## Step 5: Tuning Regularization Parameters 
param_test6 = {
    'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}

gsearch6 = GridSearchCV(estimator=XGBClassifier(
    learning_rate=lr, 
    n_estimators=ne, 
    max_depth=md,
    min_child_weight=mcw, 
    gamma=g, 
    subsample=s, 
    colsample_bytree=cb,
    objective= 'binary:logistic', 
    scale_pos_weight=1, 
    seed=2017), 
    param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch6.fit(dtrain[predictors], dtrain[target])
gsearch6.best_params_, gsearch6.best_score_

ra = 
#({'reg_alpha': 1}, 0.69681910579215622)

param_test7 = {
    'reg_alpha':[0, 0.1, 0.5, 1, 5, 10]
}

gsearch7 = GridSearchCV(estimator=XGBClassifier(
    learning_rate=0.1, 
    n_estimators=68, 
    max_depth=3,
    min_child_weight=2, 
    gamma=0, 
    subsample=0.5, 
    colsample_bytree=0.7,
    objective= 'binary:logistic', 
    scale_pos_weight=1, 
    seed=2017), 
    param_grid = param_test7, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch7.fit(dtrain[predictors], dtrain[target])
gsearch7.best_params_, gsearch7.best_score_

# 

xgb3 = XGBClassifier(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=md,
    min_child_weight=mcw,
    gamma=g,
    subsample=s,
    colsample_bytree=cb,
    reg_alpha=ra,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=2017)
modelfit(xgb3, dtrain, dtest, predictors)

#87

## Step 6: Reducing Learning Rate

xgb4 = XGBClassifier(
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
modelfit(xgb4, dtrain, dtest, predictors)



######################################################################################
# 4: Model Building
######################################################################################
print('Initializing xgboost.sklearn.XGBClassifier and starting training...')

st = datetime.now()

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
modelfit(xgbFinal, dtrain, dtest, predictors)

print(datetime.now()-st)


 