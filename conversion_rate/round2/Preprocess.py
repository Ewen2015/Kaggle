#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import pandas as pd
from sklearn import preprocessing
import warnings

warnings.filterwarnings("ignore")

def timestamp_datetime(timestamp):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

def Preprocess(data):
    print('\tpreprocessing...')
    
    LabelEncoder = preprocessing.LabelEncoder()
    for col in ['user_id', 'item_id', 'item_brand_id', 'item_city_id', 'shop_id', 'context_id', 'context_page_id']:
        data[col] = LabelEncoder.fit_transform(data[col])

    data['realtime'] = data['context_timestamp'].apply(timestamp_datetime)
    data['realtime'] = pd.to_datetime(data['realtime'])
    data['hour'] = data['realtime'].dt.hour
    data['minute'] = data['realtime'].dt.minute

    del data['realtime']
    del data['context_timestamp']
    return data

bin_list = ['shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description']

def Bin(data):
    print('\tbinning...')
    for col in bin_list:
        data[col+'_bin'] = pd.qcut(data[col], 5, labels=False, duplicates='drop')
    return data

def base_counter(data, Base, List, count_base=True, delete_base=True):
    if count_base:
        itemCount = data.groupby([Base], as_index=False)['instance_id'].agg({Base + '_cnt': 'count'})
        data = data.merge(itemCount, on=Base, how='left')
    for f in List:
        print('\t\t\ton ', f)
        itemCount = data.groupby([Base, f], as_index=False)['instance_id'].agg({str(f) + '_' + Base + '_cnt': 'count'})
        data = data.merge(itemCount, on=[Base, f], how='left')
        data[str(f)+'_'+Base+'_prob'] = (data[str(f)+'_'+Base+'_cnt']/data[Base+'_cnt']).round(3)
        del data[str(f) + '_' + Base + '_cnt']
    if delete_base:
        data[Base+'_ratio'] = (data[Base + '_cnt']/data.shape[0]).round(6)
        del data[Base + '_cnt']
    return data

user_list = ['user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']
item_list = ['item_id', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']
shop_list = ['shop_id', 'shop_review_num_level', 'shop_star_level', 'shop_review_positive_rate_bin', 'shop_score_service_bin', 'shop_score_delivery_bin', 'shop_score_description_bin']
context_list = ['context_id', 'hour', 'context_page_id']

def UserItemShopContext(data):
    print('\tUserItemShopContext starts...')
    for i in range(len(user_list)):
        print('\t\tuser ', user_list[i])                                                         
        data = data.pipe(base_counter, Base=user_list[i], List=item_list, delete_base=False)                       # user-item
        data = data.pipe(base_counter, Base=user_list[i], List=shop_list, count_base=False, delete_base=False)     # user-shop
        data = data.pipe(base_counter, Base=user_list[i], List=context_list, count_base=False, delete_base=True)   # user-context       
    for i in range(len(item_list)):
        print('\t\titem ', item_list[i])                                                       
        data = data.pipe(base_counter, Base=item_list[i], List=user_list)                        # item-user
    for i in range(len(shop_list)):
        print('\t\tshop ', shop_list[i])                                                         
        data = data.pipe(base_counter, Base=shop_list[i], List=user_list)                        # shop-user
    for i in range(len(context_list)):
        print('\t\tcontext ', context_list[i])                                                         
        data = data.pipe(base_counter, Base=context_list[i], List=user_list)                     # context-user 
    return data

if __name__ == "__main__":
    wd = '/Users/ewenwang/Documents/practice_data/conversion_rate/'
    file = ['round2_train_7.txt', 'round2_test.txt']

    output = ['round2_processed_train_7.txt', 'round2_processed_test.txt']

    # train = pd.read_csv(wd+file[0], sep=' ')
    # print('training data')
    # processed_train = (train.pipe(Preprocess)
    #                         .pipe(Bin)
    #                         .pipe(UserItemShopContext))
    # print('saving...')
    # processed_train.to_csv(wd+output[0], sep=' ', index=False)

    test = pd.read_csv(wd+file[1], sep=' ')
    print('test data')
    processed_test = (test.pipe(Preprocess)
                          .pipe(Bin)
                          .pipe(UserItemShopContext))  
    print('saving...')
    processed_test.to_csv(wd+output[1], sep=' ', index=False)