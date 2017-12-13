'''
@author: Ewen Wang

Introduction:

The competition asks you to determine whether a loan will default, as well as 
the loss incurred if it does default. 

To simplify the problem, we divide the question into two stages:

1) Classification: default loan
2) Regression: the amount of loss

This code has finished the feature engineering with the code of guocong, which will 
be imported from features. You may develop your own way to do feature engineering as 
well. 

You need to use your own function to get an estimator to do classification. 

A report function has beed provided as well. Feel free to use it. 
'''
import os
import pandas as pd
import numpy as np
from datetime import datetime
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
import features
import warnings
warnings.filterwarnings("ignore")

read_csv = partial(pd.read_csv, na_values=['NA', 'na'], low_memory=False)


def loadData(path, filename, test_size=0.33, seed=2017):
	print('\nLoading data...')
	dataset = read_csv(os.path.join(path, filename))
	train, test = train_test_split(
		dataset, test_size=test_size, random_state=seed)
	test_y = pd.DataFrame(test['loss'].values, columns=['loss'])
	train_y = pd.DataFrame(train['loss'].values, columns=['loss'])
	test_y[test_y>0] = 1
	train_y[train_y>0] = 1
	print('\nFeature engineering...')
	test_X = pd.DataFrame(features.FeatureSelector().fit_transform(test),
						  columns=["Var %d" % (i + 1) for i in range(435)])
	train_X = pd.DataFrame(features.FeatureSelector().fit_transform(train),
						   columns=["Var %d" % (i + 1) for i in range(435)])
	dtrain, dtest = train_y.join(train_X), test_y.join(test_X)
	return dtrain, dtest


'''
Note: Define your algorithm to find the optimal estimator here.

input: dtrain
output: estimator
'''

def yourAlgorithm(Train):
	estimator = 'This is a fake estimator!'
	print('\nYou got a fake estimator!')
	return estimator


def report(clf, Train, Test, predictors, target):
	print('predicting...')
	dtrain_predictions = clf.predict(Train[predictors])
	dtest_predictions = clf.predict(Test[predictors])
	dtrain_predprob = clf.predict_proba(Train[predictors])
	dtest_predprob = clf.predict_proba(Test[predictors])
	print("\nModel Report")
	print("Accuracy : %f" % metrics.accuracy_score(
		dtrain[target], dtrain_predictions))
	print("AUC Score (Train): %f" %
		  metrics.roc_auc_score(dtrain[target], dtrain_predprob))
	print('AUC Score (Test): %f' %
		  metrics.roc_auc_score(dtest[target], dtest_predprob))
	print(classification_report(dtest[target], dtest_predictions))
	return None


def main():
	print('\nLoading and feature engineering may take you 45 seconds, feel free to take a break!')
	path = '/Users/ewenwang/Documents/loan'
	filename = 'train_v2.csv'
	st = datetime.now()
	dtrain, dtest = loadData(path, filename)
	print('\nActually, ', datetime.now()-st)
	
	optimized_clf = yourAlgorithm(dtrain)

	print()
	# # Use following code to predict and report.
	# target = 'loss'
	# predictors = [x for x in dtrain.columns if x not in [target]]
	# report(optimized_clf, dtrain, dtest, predictors, target)
	return None

if __name__ == '__main__':

	main()