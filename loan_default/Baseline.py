#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wang.enqun@outlook.com
license:    Apache License 2.0
"""

class Baseline(object):
    """Provide general machine learning models as baseline."""
    def __init__(self, train, valid, target, features, impute=True):
        super(Baseline, self).__init__()
        self.target = target
        self.features = features

        self.train = train
        self.valid = valid
        
        if impute:
            import pandas as pd
            from sklearn.preprocessing import Imputer

            self.train_prep = pd.DataFrame(Imputer(strategy='mean').fit_transform(self.train), columns=self.train.columns)
            self.valid_prep = pd.DataFrame(Imputer(strategy='mean').fit_transform(self.valid), columns=self.valid.columns)
        else:
            self.train_prep = self.train
            self.valid_prep = self.valid          

    def LR(self, report=False):
        """Logistic Regression.

        Args:
            feature_num: number of feaures to keep in the model.
            report: whether print out the model analysis report.
        Returns:
            Logistic regression model."""
        from sklearn.linear_model import LogisticRegression

        self.lr = LogisticRegression(n_jobs=-1)
        self.lr.fit(self.train_prep[self.features], self.train_prep[self.target])

        if report:
            from Report import Report
            rpt = Report(self.lr, self.train_prep, self.valid_prep, self.target, self.features)
            rpt.ALL()

        return self.lr
    
    def RF(self, report=False):
        """Random Forest.

        Args:
            report: whether print out the model analysis report.
        Returns:
            Decision tree model generated from Random Forest."""
        from sklearn.ensemble import RandomForestClassifier

        self.rf = RandomForestClassifier(n_estimators=1000, 
                                        max_features='sqrt',
                                        max_depth=10,
                                        random_state=0, 
                                        n_jobs=-1)
        self.rf.fit(self.train_prep[self.features], self.train_prep[self.target])

        if report:
            from Report import Report
            rpt = Report(self.rf, self.train_prep, self.valid_prep, self.target, self.features)
            rpt.ALL()

        return self.rf

    def GBDT(self, report=False):
        """Gradient Boosting Decision Tree.

        Args:
            report: whether print out the model analysis report.
        Returns:
            Decision tree model generated from Gradient Boosting Decision Tree."""
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(self.train, test_size=0.2, random_state=0)

        lgb_train = lgb.Dataset(train[self.features], train[self.target], free_raw_data=False)
        lgb_valid = lgb.Dataset(test[self.features], test[self.target], reference=lgb_train, free_raw_data=False)
        
        params = {
            'boosting_type': 'gbdt',
            'objective': 'bianry',
            'metric': 'auc',
            'num_leaves': 64,
            'learning_rate': 0.01,
            'feature_fraction': 0.75,
            'bagging_fraction': 0.75,
            'bagging_freq': 5,
            'verbose': 0
        }

        self.gbdt = lgb.train(params,
                        lgb_train,
                        num_boost_round=10000,
                        valid_set=lgb_valid,
                        early_stopping_round=200,
                        verbose_eval=100)
        if report:
            from Report import Report
            rpt = Report(self.gbdt, self.train, self.valid, self.target, self.features)
            rpt.ALL()

        return self.gbdt

    def NN(self, report=False):
        """Neutral Network.

        Args:
            report: whether print out the model analysis report.
        Returns:
            One layer neutral network model."""
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.wrappers.scikit_learn import KerasClassifier

        def baseline_model():
            model = Sequential()
            model.add(Dense(8, input_dim=len(self.features), activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model    

        self.nn = KerasClassifier(build_fn=baseline_model, epochs=5, batch_size=5, verbose=1)
        self.nn.fit(self.train[self.features], self.train[self.target])

        if report:
            from Report import Report
            rpt = Report(self.nn, self.train, self.valid, self.target, self.features)
            rpt.ALL()

        return self.nn