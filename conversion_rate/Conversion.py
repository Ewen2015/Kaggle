import pandas as pd
import time
import lightgbm as lgb 
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
import gossipcat as gc

wd = '/Users/ewenwang/Documents/practice_data/conversion_rate/'
raw_data = ['round1_ijcai_18_train_20180301.txt', 'round1_ijcai_18_test_a_20180301.txt']
new_data = ['train.txt', 'test.txt']

def load_data(wd=wd, data_list=raw_data):
    return pd.read_csv(wd+data_list[0], sep=' '), pd.read_csv(wd+data_list[1], sep=' ')

def save_data(data, file):
    data.to_csv(wd+file, index=False, sep=' ')
    return None

def timestamp_datetime(timestamp):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

def Preprocess(data):

    for i in range(3):
        data['category_%d'%(i)] = data['item_category_list'].apply(lambda x: x.split(";")[i] if len(x.split(";"))>i else " ")
    del data['item_category_list']
    del data['category_0']

    for i in range(3):
        data['property_%d'%(i)] = data['item_property_list'].apply(lambda x: x.split(";")[i] if len(x.split(";"))>i else " ")
    del data['item_property_list']

    for i in range(3):
        data['predict_category_%d'%(i)] = data['predict_category_property'].apply(lambda x: str(x.split(";")[i]).split(":")[0] if len(x.split(";"))>i else " ")
    del data['predict_category_property']

    data['time'] = data.context_timestamp.apply(timestamp_datetime)
    del data['context_timestamp']
    data['time_tmp'] = pd.to_datetime(data['time'])
    data['day'] = data['time_tmp'].dt.day
    data['week'] = data['time_tmp'].dt.weekday
    data['hour'] = data['time_tmp'].dt.hour
    del data['time_tmp']
    user_query_day = data.groupby(['user_id', 'day']).size().reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left',on=['user_id', 'day', 'hour'])

    return data


target = 'is_trade'

drop_list = ['instance_id', 'time', 'day']

features = ['item_id', 'item_brand_id', 'item_city_id', 
'user_id', 'user_gender_id', 'user_occupation_id', 
'context_id', 'context_page_id', 
'shop_id', 
'category_1', 'category_2', 'property_0', 'property_1', 'property_2', 
'predict_category_0', 'predict_category_1', 'predict_category_2',

'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 
'user_age_level', 'user_star_level', 
'shop_score_service', 'shop_score_delivery', 'shop_score_description', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 
'week', 'hour', 'user_query_day', 'user_query_day_hour']

id_features = ['item_id', 'item_brand_id', 'item_city_id', 
'user_id', 'user_gender_id', 'user_occupation_id', 
'context_id', 'context_page_id', 
'shop_id', ]

numeric_features = ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 
'user_age_level', 'user_star_level', 
'shop_score_service', 'shop_score_delivery', 'shop_score_description', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 
'week', 'hour', 'user_query_day', 'user_query_day_hour']

categorical_features = ['item_id', 'item_brand_id', 'item_city_id', 
'user_id', 'user_gender_id', 'user_occupation_id', 
'context_id', 'context_page_id', 
'shop_id', 
'category_1', 'category_2', 'property_0', 'property_1', 'property_2', 
'predict_category_0', 'predict_category_1', 'predict_category_2']


params = {
    'boosting_type': 'gbdt', 
    'num_leaves': 31, 
    'max_depth': -1, 
    'learning_rate': 0.01,       #
    'n_estimators': 100, 
    'subsample_for_bin': 200000, 
    'objective': None, 
    'class_weight': None, 
    'min_split_gain': 0.0, 
    'min_child_weight': 0.001, 
    'min_child_samples': 20, 
    'subsample': 0.75,          #
    'subsample_freq': 1, 
    'colsample_bytree': 0.75,   #
    'reg_alpha': 0.0, 
    'reg_lambda': 0.0, 
    'random_state': 2018,       #
    'n_jobs': -1, 
    'silent': True
}

def commom(train, test, category_features):
    results = pd.DataFrame(columns=['column', 'count_test', 'count_train', 'count_common', 'pct_test', 'pct_train'])
    for ind, val in enumerate(category_features):
        df = pd.DataFrame()
        df.loc[ind, 'column']  = val
        df.loc[ind, 'count_test'] = len(test[val].unique())
        df.loc[ind, 'count_train']  = len(train[val].unique())
        df.loc[ind, 'count_common']  = pd.merge(test, train, how='inner', on=[val]).groupby(val).size().shape[0]
        df.loc[ind, 'pct_test'] = df.loc[ind, 'count_common']/df.loc[ind, 'count_test']
        df.loc[ind, 'pct_train'] = df.loc[ind, 'count_common']/df.loc[ind, 'count_train']
        results = results.append(df)
    results.sort_values(by=['pct_test'], ascending=False)
    
    return results

class Train(object):
    """docstring for Train"""
    def __init__(self, train, test, target, features, validation_day=24):
        super(Train, self).__init__()
        self.train = train
        self.test = test
        self.target = target
        self.features = features

        self.dtrain = self.train.loc[self.train.day<validation_day]
        self.validation = self.train.loc[self.train.day==validation_day]

        self.pred_test = []
        
    def Baseline(self):
        clf = lgb.LGBMClassifier(
            boosting_type='gbdt', 
            num_leaves=31, 
            max_depth=-1, 
            learning_rate=0.01,         #
            n_estimators=500,           #
            subsample_for_bin=200000, 
            objective=None, 
            class_weight=None, 
            min_split_gain=0.0, 
            min_child_weight=0.001, 
            min_child_samples=20, 
            subsample=0.75,             #
            subsample_freq=1, 
            colsample_bytree=0.75,      #
            reg_alpha=0.0, 
            reg_lambda=0.0, 
            random_state=2018,          #
            n_jobs=-1, 
            silent=True)

        clf.fit(self.dtrain[self.features], self.dtrain[self.target], feature_name=self.features)
        self.validation['predicted_score'] = clf.predict_proba(self.validation[self.features])[:, 1]
        print('log loss: ', log_loss(self.validation[self.target], self.validation['predicted_score']))

        return clf

    def simAnneal(self, alpha=0.75, scoring='neg_log_loss', n_jobs=1, results=True, seed=2018):
        """ Hyper parameter tuning with simulated annealing.

        Employes the simulated annealing to find the optimal hyper parameters and 
        return an optimized classifier.

        Args:
            train: A training set of your machine learning project.
            target: The target variablet; limited to binary.
            predictors: The predictors.
            results: Whether print the progress out; default with True.

        Returns:
            A classifier generated from gbdt with simulated annealing hyper parameter tuning.
        """
        params = {
            'max_depth': [i for i in range(1, 21, 1)],
            'subsample': [i / 100.0 for i in range(1, 101, 1)],
            'colsample_bytree': [i / 100.0 for i in range(1, 101, 1)],
        }

        print('simulating...')

        gbm = LGBMClassifier(
            boosting_type='gbdt', 
            num_leaves=31, 
            max_depth=-1, 
            learning_rate=0.01,         #
            n_estimators=1000,           #
            subsample_for_bin=200000, 
            objective=None, 
            class_weight=None, 
            min_split_gain=0.0, 
            min_child_weight=0.001, 
            min_child_samples=20, 
            subsample=0.75,             #
            subsample_freq=1, 
            colsample_bytree=0.75,      #
            reg_alpha=0.0, 
            reg_lambda=0.0, 
            random_state=2018,          #
            n_jobs=-1, 
            silent=True
        )
        sa = gc.SimulatedAnneal(
            gbm, 
            params, 
            scoring=scoring, 
            T=10.0, 
            T_min=0.001, 
            alpha=alpha,
            n_trans=5, 
            max_iter=0.25, 
            max_runtime=300, 
            cv=5, 
            random_state=seed, 
            verbose=True, 
            refit=True, 
            n_jobs=n_jobs)
        
        sa.fit(self.dtrain[self.features], self.dtrain[self.target])
        self.validation['predicted_score'] = sa.best_estimator_.predict_proba(self.validation[self.features])[:, 1]

        if results:
            print('\nbest score:', '{:.6f}'.format(sa.best_score_),
                  '\nlog loss:', '{:.6f}'.format(log_loss(self.validation[self.target], self.validation['predicted_score'])),
                  '\nbest parameters:', str({key: '{:.2f}'.format(value) for key, value in sa.best_params_.items()}))

        return sa


if __name__ == '__main__':
    pass










