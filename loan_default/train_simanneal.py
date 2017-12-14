'''
@author: Ewen Wang
'''
import os
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
import SimAnneal
import models
import warnings
warnings.filterwarnings("ignore")

read_csv = partial(pd.read_csv, na_values=['NA', 'na'], low_memory=False)

param_1 = {
    # 'boosting_type': ['gbdt', 'dart'],
    # 'num_leaves': [i for i in range(3, 20, 1)],
    # 'max_depth': [i for i in range(1, 5, 1)],
    'subsample': [i / 10.0 for i in range(1, 10, 1)],
    'colsample_bytree': [i / 10.0 for i in range(1, 10, 1)],
}

# 0.997932976106
# {'max_depth': 4, 'subsample': 0.69999999999999996, 'colsample_bytree': 0.69999999999999996}


param_2 = {
    #     'boosting_type': ['gbdt', 'dart'],
    'num_leaves': [i for i in range(3, 20, 1)],
    'max_depth': [3, 4, 5],
    'subsample': [i / 100.0 for i in range(65, 75, 1)],
    'colsample_bytree': [i / 100.0 for i in range(65, 75, 1)],
}

# 0.998018198406
# {'num_leaves': 9, 'max_depth': 5, 'subsample': 0.71999999999999997, 'colsample_bytree': 0.71999999999999997}

param_3 = {
    #     'boosting_type': ['gbdt', 'dart'],
    'num_leaves': [i for i in range(7, 15, 1)],
    'max_depth': [4, 5, 6],
    'subsample': [i / 1000.0 for i in range(715, 725, 1)],
    'colsample_bytree': [i / 1000.0 for i in range(715, 725, 1)],
}

# best score:  0.9984877073
# best parameters:  {'num_leaves': 14, 'max_depth': 5, 'subsample':
# 0.71899999999999997, 'colsample_bytree': 0.71799999999999997}

param_4 = {
    #     'boosting_type': ['gbdt', 'dart'],
    'num_leaves': [i for i in range(14, 20, 1)],
    'max_depth': [4, 5, 6],
    'subsample': [i / 10000.0 for i in range(7185, 7195, 1)],
    'colsample_bytree': [i / 10000.0 for i in range(7175, 7185, 1)],
}

# best score:  0.998516951324
# best parameters:  {'num_leaves': 18, 'max_depth': 6, 'subsample':
# 0.71930000000000005, 'colsample_bytree': 0.71779999999999999}


def loadData(path, filename, test_size=0.33, seed=2017):
    print('\nloading...')
    dataset = read_csv(os.path.join(path, filename))
    train, test = train_test_split(
        dataset, test_size=test_size, random_state=seed)
    test_y = pd.DataFrame(test['loss'].values, columns=['loss'])
    train_y = pd.DataFrame(train['loss'].values, columns=['loss'])
    test_y[test_y > 0] = 1
    train_y[train_y > 0] = 1
    test_X = pd.DataFrame(models.FeatureSelector().fit_transform(test),
                          columns=["Var %d" % (i + 1) for i in range(435)])
    train_X = pd.DataFrame(models.FeatureSelector().fit_transform(train),
                           columns=["Var %d" % (i + 1) for i in range(435)])
    dtrain, dtest = train_y.join(train_X), test_y.join(test_X)
    return dtrain, dtest


def simAnneal(Train, predictors, target, param, results=True):
    print('\nsimulating...')
    gbm = LGBMClassifier(
        learning_rate=0.01, n_estimators=5000, objective='binary', metric='auc',
        save_binary=True, is_unbalance=True, random_state=2017
    )
    sa = SimAnneal.SimulatedAnneal(gbm, param, T=10.0, T_min=0.001, alpha=0.75,
                                   verbose=True, max_iter=0.25, n_trans=5, max_runtime=300,
                                   cv=3, scoring='roc_auc', refit=True)
    sa.fit(Train[predictors].as_matrix(), Train[target].as_matrix())
    if results:
        print('\n best score: ', sa.best_score_,
              '\n best parameters: ', sa.best_params_)
    optimized_clf = sa.best_estimator_
    return optimized_clf


def report(clf, Train, Test, predictors, target):
    print('\npredicting...')
    dtrain_predictions = clf.predict(Train[predictors])
    dtest_predictions = clf.predict(Test[predictors])
    dtrain_predprob = clf.predict_proba(Train[predictors])[:, 1]
    dtest_predprob = clf.predict_proba(Test[predictors])[:, 1]
    print("\nModel Report")
    print("Accuracy : %f" % metrics.accuracy_score(
        Train[target], dtrain_predictions))
    print("AUC Score (Train): %f" %
          metrics.roc_auc_score(Train[target], dtrain_predprob))
    print('AUC Score (Test): %f' %
          metrics.roc_auc_score(Test[target], dtest_predprob))
    print(classification_report(Test[target], dtest_predictions))
    return None


def main():
    path = '/Users/ewenwang/Documents/practice_data'
    filename = 'loan_default.csv'
    dtrain, dtest = loadData(path, filename)
    target = 'loss'
    predictors = [x for x in dtrain.columns if x not in [target]]
    optimized_clf = simAnneal(dtrain, predictors, target, param=param_1)
    report(optimized_clf, dtrain, dtest, predictors, target)
    return None

if __name__ == '__main__':

    main()
