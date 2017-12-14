import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from simulated_annealing.optimize import SimulatedAnneal

if __name__ == '__main__':
	print('\nLoading data...')
	wd = '/Users/ewenwang/Downloads'

	DataFileX, DataFileY = "train_x.txt", "train_y.txt"
	X = pd.read_csv(DataFileX, low_memory=False).as_matrix()
	y = pd.read_csv(DataFileY, low_memory=False).as_matrix()

	print('\nSpliting the data into test and train sets...')
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=2017)

	print('\nSetting hypyerparameters...')
	param_test = {
		'boosting_type': ['gbdt', 'dart'],
		'num_leaves': [i for i in range(3, 20, 1)],
		'max_depth': list(range(1, 5, 1)),
		'subsample': [i / 100.0 for i in range(20, 90, 1)],
		'colsample_bytree': [i / 100.0 for i in range(20, 90, 1)],
	}

	print('\nSetting LGBMClassifier...')
	gbm = LGBMClassifier(
		learning_rate=0.01,
		n_estimators=5000,
		objective='binary',
		save_binary=True,
		is_unbalance=True,
		seed=2)

	print('\nInitializing Simulated Annealing and fitting...')
	sa = SimulatedAnneal(gbm, param_test, T=10.0, T_min=0.001, alpha=0.75,
						 verbose=True, max_iter=0.25, n_trans=5, max_runtime=300,
						 cv=5, scoring='roc_auc', refit=True)
	sa.fit(X_train, y_train)

	print('\nPrinting the best score and the best params...')
	print(sa.best_score_, sa.best_params_)

	print('\nUsing the best estimator to predict...')
	optimized_gbm = sa.best_estimator_

	y_train_pred = optimized_gbm.predict(X_train)
	y_train_prob = optimized_gbm.predict_proba(X_train)[:,1]
	y_test_pred = optimized_gbm.predict(X_test)
	y_test_prob = optimized_gbm.predict_proba(X_test)[:,1]
	   
	print("\nModel Report")

	print('\nPrinting a report of precision, recall, f1_score...')
	print(classification_report(y_test, y_test_pred))

	print('\nPrinting a report of accuracy and AUC...')
	print("Accuracy : %.4g" % metrics.accuracy_score(y_train, y_train_pred))
	print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, y_train_prob)) 
	print('AUC Score (Test): %f' % metrics.roc_auc_score(y_test, y_test_prob))



