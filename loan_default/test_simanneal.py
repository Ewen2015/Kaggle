'''
@author: Ewen Wang

Model Report
Accuracy : 1
AUC Score (Train): 1.000000
AUC Score (Test): 0.998815
             precision    recall  f1-score   support

          0       1.00      0.99      0.99     31630
          1       0.92      0.97      0.95      3176

avg / total       0.99      0.99      0.99     34806
'''
import os
import pandas as pd
import numpy as np 
from datetime import datetime
from lightgbm import LGBMClassifier
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
import features
import warnings
warnings.filterwarnings("ignore")

read_csv = partial(pd.read_csv, na_values=['NA', 'na'], low_memory=False)

lgbm = LGBMClassifier(
	boosting_type='gbdt', 
	num_leaves=18, 
	max_depth=6, 
	learning_rate=0.01, 
	n_estimators=5000, 
	objective='binary', 
	subsample=0.7193, 
	colsample_bytree=0.7178, 
	random_state=2017
	)

def loadData(path, filename, test_size=0.33, seed=2017):
	print('\nloading...')
	print('\nthis may take you 1 minute...')
	dataset = read_csv(os.path.join(path, filename))
	train, test = train_test_split(dataset, test_size=test_size, random_state=seed)
	test_y = pd.DataFrame(test['loss'].values, columns=['loss'])
	train_y = pd.DataFrame(train['loss'].values, columns=['loss'])
	test_y[test_y>0] = 1
	train_y[train_y>0] = 1
	test_X = pd.DataFrame(features.FeatureSelector().fit_transform(test),
						  columns=["Var %d" % (i + 1) for i in range(435)])
	train_X = pd.DataFrame(features.FeatureSelector().fit_transform(train),
						   columns=["Var %d" % (i + 1) for i in range(435)])
	dtrain, dtest = train_y.join(train_X), test_y.join(test_X)
	return dtrain, dtest

def modelfit(lgbm, dtrain, dtest, target, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
	print('\nfitting...') 
	print('\nthis may take you 5 minutes...')
	if useTrainCV:
		lgb_param = lgbm.get_params()
		lgbtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values)
		lgbtest = lgb.Dataset(dtest[predictors].values, label=dtest[target].values, reference=lgbtrain)
		cvresult = lgb.cv(lgb_param, 
						  lgbtrain, 
						  num_boost_round=lgbm.get_params()['n_estimators'], 
						  nfold=cv_folds,
						  metrics='auc', 
						  early_stopping_rounds=early_stopping_rounds)
		cv = pd.DataFrame(cvresult)
		lgbm.set_params(n_estimators=cv.shape[0])
		print(cv.tail(10))
	lgbm.fit(dtrain[predictors], dtrain[target], eval_metric='auc')
	dtrain_predictions = lgbm.predict(dtrain[predictors])
	dtest_predictions = lgbm.predict(dtest[predictors])
	dtrain_predprob = lgbm.predict_proba(dtrain[predictors])[:,1]
	dtest_predprob = lgbm.predict_proba(dtest[predictors])[:,1]
	print("\nModel Report")
	print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target], dtrain_predictions))
	print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob)) 
	print('AUC Score (Test): %f' % metrics.roc_auc_score(dtest[target], dtest_predprob))
	print(classification_report(dtest[target], dtest_predictions))
	# lgb.plot_importance(lgbm, figsize=(12, 16), grid=False)
	return None

def main():
	print('\nThe whole programming may run for 6 minutes, free feel to take a break!')
	path = '/Users/ewenwang/Documents/loan'
	filename = 'train_v2.csv'
	st = datetime.now()
	dtrain, dtest = loadData(path, filename)
	print(datetime.now()-st)
	target = 'loss'
	predictors = [x for x in dtrain.columns if x not in [target]]
	st = datetime.now()
	modelfit(lgbm, dtrain, dtest, target, predictors, useTrainCV=False)
	print(datetime.now()-st)
	print('\nDone!')
	return None

if __name__ == '__main__':
	main()


