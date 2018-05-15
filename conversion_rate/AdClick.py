#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     SnowColdplay
address:    https://github.com/SnowColdplay/almm_baseline/blob/master/base0326SUB.py
revised by: Ewen Wang
"""
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
        data['item_property_list' + str(i)] = LabelEncoder.fit_transform(data['item_property_list'].map(lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))

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


# feature generation
def divider_generator(bin_1_s=None, bin_1_e=None, bin_2_e=None):
    def divider(x):
        if (x >= bin_1_s) & (x <= bin_1_e):
            return 1
        elif (x >= (bin_1_e + 1)) & (x <= bin_2_e):
            return 2
        else:
            return 3
    return divider

map_hour = divider_generator(7, 12, 20)
deliver1 = divider_generator(2, 4, 7)
review1 = divider_generator(2, 12, 15)
service1 = divider_generator(2, 7, 9)
describe1 = divider_generator(2, 8, 10)

def binner_generator(Bin=None, Range=None, stop=None, expt=None):
    def binner(x):
        Bin = 0.1
        for i in range(1, Range):
            if (x >= stop + Bin * (i - 1)) & (x <= stop + Bin * i):
                return i + 1
        if x == expt:
            return 1
    return binner

deliver = binner_generator(0.1, 20, 4.1, -5)
review = binner_generator(0.02, 30, 0.714, -1)
service = binner_generator(0.1, 20, 3.93, -1)
describe = binner_generator(0.1, 30, 3.93, -1)

def DayDivider(data):
    print('day deviding...')
    data['hour_map'] = data['hour'].apply(map_hour)
    return data

def ShopDivider(data):
    print('shop deviding...')

    data['shop_score_delivery'] = data['shop_score_delivery'] * 5
    data = data[data['shop_score_delivery'] != -5]
    data['deliver_map'] = data['shop_score_delivery'].apply(deliver)
    data['deliver_map'] = data['deliver_map'].apply(deliver1)

    data['shop_score_service'] = data['shop_score_service'] * 5
    data = data[data['shop_score_service'] != -5]
    data['service_map'] = data['shop_score_service'].apply(service)
    data['service_map'] = data['service_map'].apply(service1)

    data['shop_score_description'] = data['shop_score_description'] * 5
    data = data[data['shop_score_description'] != -5]
    data['de_map'] = data['shop_score_description'].apply(describe)
    data['de_map'] = data['de_map'].apply(describe1)

    data = data[data['shop_review_positive_rate'] != -1]
    data['review_map'] = data['shop_review_positive_rate'].apply(review)
    data['review_map'] = data['review_map'].apply(review1)

    data['normal_shop'] = data.apply(lambda x: 1 if (x.deliver_map == 3) &
                                                    (x.service_map == 3) &
                                                    (x.de_map == 3) &
                                                    (x.review_map == 3) else 0, axis=1)
    del data['de_map']
    del data['service_map']
    del data['deliver_map']
    del data['review_map']
    return data


def slider(data, days_ahead=1, all_ahead=False):
    if all_ahead:
        days_ahead = 1
    for d in range(18+days_ahead, 26):
        if all_ahead:
            df_feature = data[data['day'] < d]
        else:
            df_feature = data[(data['day'] >= d - days_ahead) & (data['day'] < d)]
        
        df_test = data[data['day'] == d]
        user_cnt = df_feature.groupby(by='user_id').count()['instance_id'].to_dict()
        item_cnt = df_feature.groupby(by='item_id').count()['instance_id'].to_dict()
        shop_cnt = df_feature.groupby(by='shop_id').count()['instance_id'].to_dict()
        df_test['user_cnt_'+str(days_ahead)+'_ahead'] = df_test['user_id'].apply(lambda x: user_cnt.get(x, 0))
        df_test['item_cnt_'+str(days_ahead)+'_ahead'] = df_test['item_id'].apply(lambda x: item_cnt.get(x, 0))
        df_test['shop_cnt_'+str(days_ahead)+'_ahead'] = df_test['shop_id'].apply(lambda x: shop_cnt.get(x, 0))

        df_test = df_test[['user_cnt_'+str(days_ahead)+'_ahead', 'item_cnt_'+str(days_ahead)+'_ahead', 'shop_cnt_'+str(days_ahead)+'_ahead', 'instance_id']]
        
        if d == 18+days_ahead:
            DF_test = df_test
        else:
            DF_test = pd.concat([df_test, DF_test])
    data = pd.merge(data, DF_test, on=['instance_id'], how='left')
    return data

def SlideCounter(data):
    print('sliding counting...')

    data = (data.pipe(slider, days_ahead=1, all_ahead=False)
                .pipe(slider, days_ahead=2, all_ahead=False)
                .pipe(slider, days_ahead=7, all_ahead=True))
    return data


def Combine(data):
    print('feature combining...')
    for col in ['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']:
        data[col] = data[col].apply(lambda x: 0 if x == -1 else x)

    tostr_list = ['item_sales_level', 'item_price_level', 'item_collected_level', 'user_gender_id', 'user_age_level',
                  'user_occupation_id', 'user_star_level', 'shop_review_num_level', 'shop_star_level']

    for col in tostr_list:
        data[col] = data[col].astype(str)

    data['sale_price'] = data['item_sales_level'] + data['item_price_level']            # item
    data['sale_collect'] = data['item_sales_level'] + data['item_collected_level']
    data['price_collect'] = data['item_price_level'] + data['item_collected_level']
    data['gender_age'] = data['user_gender_id'] + data['user_age_level']                # user
    data['gender_occ'] = data['user_gender_id'] + data['user_occupation_id']
    data['gender_star'] = data['user_gender_id'] + data['user_star_level']
    data['review_star'] = data['shop_review_num_level'] + data['shop_star_level']       # shop

    inted_list = ['item_sales_level', 'item_price_level', 'item_collected_level', 'user_gender_id', 'user_age_level',
                  'user_occupation_id', 'user_star_level', 'shop_review_num_level', 'shop_star_level', 'review_star',  'sale_price',
                  'sale_collect', 'price_collect', 'gender_age', 'gender_occ', 'gender_star']

    for col in inted_list:
        data[col] = data[col].astype(int)

    del data['review_star']
    return data


def base_counter(data, Base, List, count_base=True, delete_base=True):
    if count_base:
        itemCount = data.groupby([Base], as_index=False)['instance_id'].agg({Base + '_cnt': 'count'})
        data = data.merge(itemCount, on=Base, how='left')
    for f in List:
        itemCount = data.groupby([Base, f], as_index=False)['instance_id'].agg({str(f) + '_' + Base + '_cnt': 'count'})
        data = data.merge(itemCount, on=[Base, f], how='left')
        data[str(f) + '_' + Base + '_prob'] = data[str(f) + '_' + Base + '_cnt'] / data[Base + '_cnt']
    if delete_base:
        del data[Base + '_cnt']
    return data

user_list = ['user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']
item_list = ['item_id', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']
shop_list = ['shop_id', 'shop_review_num_level', 'shop_star_level']
context_list = ['context_id', 'hour_map', 'context_page_id']

def UserItemShopContext(data):
    print('UserItemShopContext starts...')
    for i in range(len(user_list) - 1):                                                         # user
        data = data.pipe(base_counter, Base=user_list[i], List=user_list[i + 1:])
    for i in range(len(item_list) - 1):                                                         # item
        data = data.pipe(base_counter, Base=item_list[i], List=item_list[i + 1:])
    for i in range(len(shop_list) - 1):                                                         # shop
        data = data.pipe(base_counter, Base=shop_list[i], List=shop_list[i + 1:])
    for i in range(len(context_list) - 1):                                                      # context
        data = data.pipe(base_counter, Base=context_list[i], List=context_list[i + 1:])
    for i in range(len(user_list) - 1):                                                         # user-item
        data = data.pipe(base_counter, Base=user_list[i], List=item_list, delete_base=False)
    for i in range(len(user_list) - 1):                                                         # user-shop
        data = data.pipe(base_counter, Base=user_list[i], List=shop_list, count_base=False, delete_base=False)
    for i in range(len(user_list) - 1):                                                         # user-context
        data = data.pipe(base_counter, Base=user_list[i], List=context_list, count_base=False)
    for i in range(len(item_list) - 1):                                                         # item-shop
        data = data.pipe(base_counter, Base=item_list[i], List=shop_list, delete_base=False)
    for i in range(len(item_list) - 1):                                                         # item-context
        data = data.pipe(base_counter, Base=item_list[i], List=context_list, count_base=False)
    for i in range(len(shop_list) - 1):                                                         # shop-context
        data = data.pipe(base_counter, Base=shop_list[i], List=context_list)
    return data



if __name__ == "__main__":
    wd = '/Users/ewenwang/Documents/practice_data/conversion_rate/'
    file = ['round1_ijcai_18_train_20180301.txt', 'round1_ijcai_18_test_a_20180301.txt', 'round1_ijcai_18_test_b_20180418.txt']

    train = pd.read_csv(wd+file[0], sep=" ")
    test_a = pd.read_csv(wd+file[1], sep=" ")
    test_b = pd.read_csv(wd+file[2], sep=" ")
    data = pd.concat([train, test_a, test_b])

    data = (data.drop_duplicates(subset='instance_id')
                .pipe(Preprocess)
                .pipe(DayDivider)
                .pipe(ShopDivider)
                .pipe(SlideCounter)
                .pipe(Combine)
                .pipe(UserItemShopContext))  

    data.to_csv(wd+'snow_data_UISC_union.txt', sep=' ', index=False)
