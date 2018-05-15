#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import pandas as pd
from sklearn import preprocessing
import warnings

warnings.filterwarnings("ignore")

# preprocess
def timestamp_datetime(timestamp):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

def Preprocess(data):
    print('preprocessing...')
    
    LabelEncoder = preprocessing.LabelEncoder()
    for col in ['item_id', 'item_brand_id', 'item_city_id', 'user_id', 'context_id', 'shop_id']:
        data[col] = LabelEncoder.fit_transform(data[col])

    for col in ['item_category_list', 'item_property_list', 'predict_category_property']:
        data['len_' + col] = data[col].map(lambda x: len(str(x).split(';')))

    # item
    for i in range(1, 3):   # item_category_list_1: unique value
        data['item_category_list_' + str(i)] = LabelEncoder.fit_transform(data['item_category_list'].map(lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))
    for i in range(10):
        data['item_property_list_' + str(i)] = LabelEncoder.fit_transform(data['item_property_list'].map(lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))

    # user
    data['gender_bin'] = data['user_gender_id'].apply(lambda x: 1 if x == -1 else 2)
    data['occupation_bin'] = data['user_occupation_id'].apply(lambda x: 1 if x == -1 | x == 2003 else 2)
    data['star_bin'] = data['user_star_level'].apply(lambda x: 1 if x == -1 | x == 3000 else 3 if x == 3009 | x == 3010 else 2)

    # context
    for i in range(5):
        data['predict_category_property_' + str(i)] = LabelEncoder.fit_transform(data['predict_category_property'].map(lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))
    data['realtime'] = data['context_timestamp'].apply(timestamp_datetime)
    data['realtime'] = pd.to_datetime(data['realtime'])
    data['day'] = data['realtime'].dt.day
    data['hour'] = data['realtime'].dt.hour

    # shop
    return data

if __name__ == "__main__":
    wd = '/Users/ewenwang/Documents/practice_data/conversion_rate/'
    train_file = ['round1_ijcai_18_train_20180301.txt', 'round2_train.txt']
    test_file = ['round1_ijcai_18_test_a_20180301.txt', 'round1_ijcai_18_test_b_20180418.txt', 'round2_ijcai_18_test_a_20180425.txt']

    out_put = ['round2_preprocess_0425.txt']

    train = pd.read_csv(wd+train_file[0], sep=" ")
    test = pd.read_csv(wd+test_file[2], sep=" ")
    data = pd.concat([train, test])

    data = (data.drop_duplicates(subset='instance_id')
                .pipe(Preprocess))  

    data.to_csv(wd+out_put[0], sep=' ', index=False)