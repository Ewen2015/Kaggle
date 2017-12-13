"""
@author: ewen 
@email: ewenwangsh@cn.ibm.com
"""
import os
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


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
	# load the data
	print('\nloading...')
	wd = '/Users/ewenwang/Documents/credit/data'
	os.chdir(wd)
	dataFile = 'creditcard.csv'
	dataset = pd.read_csv(dataFile, low_memory=False)

	# set target and predictors
	target = 'Class'
	predictors = [x for x in dataset.columns if x not in [target]]

	# split the data into training and test sets 
	seed = 2017
	dtrain, dtest = train_test_split(dataset, test_size=0.33, random_state=seed)

	# build the classifier
	gbm = LGBMClassifier(
		learning_rate=0.01,
		n_estimators=5000,
		objective='binary',
		metric='auc',
		max_depth=10,
		subsample=0.83,
		colsample_bytree=0.63,
		save_binary=True,
		is_unbalance=True,
		random_state=seed
	)

	# train the model
	print('\nfitting...')
	gbm.fit(dtrain[predictors], dtrain[target])

	# report
	report(gbm, dtrain, dtest, predictors, target)

	return None


if __name__ == '__main__':
	main()
