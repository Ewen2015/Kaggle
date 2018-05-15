import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier

wd = '/Users/ewenwang/Documents/practice_data/conversion_rate/'
train_file = ['round1_ijcai_18_train_20180301.txt', 'round2_train.txt']
test_file = ['round1_ijcai_18_test_a_20180301.txt', 'round1_ijcai_18_test_b_20180418.txt', 'round2_ijcai_18_test_a_20180425.txt']

out_put = ['round2_property_0425.txt']

print('loading...')
train = pd.read_csv(wd+train_file[0], sep=" ")
# test_1_a = pd.read_csv(wd+test_file[0], sep=" ")
# test_1_b = pd.read_csv(wd+test_file[1], sep=" ")
test_2_a = pd.read_csv(wd+test_file[2], sep=" ")
data = pd.concat([train, test_2_a])

def timestamp_datetime(timestamp):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

data['realtime'] = data['context_timestamp'].apply(timestamp_datetime)
data['realtime'] = pd.to_datetime(data['realtime'])
data['day'] = data['realtime'].dt.day

print('transforming...')
count_vec = TfidfVectorizer()
data_ip = count_vec.fit_transform(data['item_property_list'])

train = data[data.is_trade.notnull()]
train_index = list(train[train.day < 24].index) 
test_index = list(train[train.day == 24].index) 
ip_train = data_ip[train_index,:]
ip_test = data_ip[test_index,:]

gbm = LGBMClassifier(
	objective='binary',
	num_leaves=24,
	max_depth=3,
	learning_rate=0.1,
	seed=2018,
	colsample_bytree=0.3,
	subsample=0.8,
	n_jobs=-1,
	n_estimators=2000
	)

print('fitting...')
gbm.fit(ip_train, train.loc[train_index, 'is_trade'], eval_set=[(ip_test, train.loc[test_index, 'is_trade'])], 
		early_stopping_rounds=10)

property_df = pd.DataFrame(columns=['instance_id', 'item_property_prob'])
property_df['instance_id'] = data['instance_id']
property_df['item_property_prob'] = gbm.predict_proba(data_ip)[:, 1]

def NatureLP(data, columns):
	
	pass

print('saving...')
property_df.to_csv(wd+out_put[0], index=False, sep=' ')




