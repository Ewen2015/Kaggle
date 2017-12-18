import os
import pandas as pd
from collections import Counter


def get_action_data(fname):
    '''This function is to input data and remove unneccessary columns.'''
    data = pd.read_csv(fname, header=0)

    data['user_id'] = data['user_id'].astype(int)
    data["time"] = pd.to_datetime(data["time"])

    data["user_sku"] = data['user_id'].map(
        str) + "_" + data["sku_id"].map(str)   # add user_sku column

    data = data.drop('model_id', 1)
    data = data.drop('cate', 1)
    data = data.drop('brand', 1)

    return data


def features_today(group):
    '''This function is to obtain today features and to be called in function behavior.'''
    behavior_type = group.type.astype(int)
    type_cnt = Counter(behavior_type)

    group['browse_num_today'] = type_cnt[1]
    group['addcart_num_today'] = type_cnt[2]
    group['delcart_num_today'] = type_cnt[3]
    group['buy_num_today'] = type_cnt[4]
    group['favor_num_today'] = type_cnt[5]
    group['click_num_today'] = type_cnt[6]

    return group[['user_sku',
                  'browse_num_today', 'addcart_num_today', 'delcart_num_today',
                  'buy_num_today', 'favor_num_today', 'click_num_today']]


def features_past(group):
    '''This function is to obtain past features and to be called in function behavior.'''
    behavior_type = group.type.astype(int)
    type_cnt = Counter(behavior_type)

    group['browse_num_past'] = type_cnt[1]
    group['addcart_num_past'] = type_cnt[2]
    group['delcart_num_past'] = type_cnt[3]
    group['buy_num_past'] = type_cnt[4]
    group['favor_num_past'] = type_cnt[5]
    group['click_num_past'] = type_cnt[6]

    return group[['user_sku',
                  'browse_num_past', 'addcart_num_past', 'delcart_num_past',
                  'buy_num_past', 'favor_num_past', 'click_num_past']]


def dependent(group):
    '''This function is to obtain dependent variable and to be called in function behavior.'''
    behavior_type = group.type.astype(int)
    type_cnt = Counter(behavior_type)

    if type_cnt[4] == 0:
        group['buy'] = 0
    else:
        group['buy'] = 1

    return group[['user_sku', 'buy']]


def behavior(data, date_start, date_end, behavior_function):
    '''This function is to obtain behavior features or dependent varibles of each
    user-product pair in a certain period.'''
    mask = (data['time'] > date_start) & (data['time'] < date_end)
    sub_data = data.loc[mask]

    user_behavior = sub_data.groupby(['user_sku'], as_index=False).apply(behavior_function)
    user_behavior = user_behavior.drop_duplicates('user_sku')

    return user_behavior


# Note: merge_data function has not inlcuded user_table and item_table yet.

def merge_data(past, today, dependent):
    '''This function is to merge features and dependent and return a train data set.'''
    train = pd.merge(behavior_today, behavior_past, on=['user_sku'], how='outer')
    train = pd.merge(dependent, train, on=['user_sku'], how='right')

    train['user_id'] = train.user_sku.apply(lambda x: x.rsplit('_', 1)[0]).astype(int)
    train['sku_id'] = train.user_sku.apply(lambda x: x.rsplit('_', 1)[1]).astype(int)

    train = pd.merge(train, user_table, on=['user_id'], how='left')
    train = pd.merge(train, item_table, on=['item_id'], how='left')

    train = train.fillna(0)

    return train


if __name__ == "__main__":

    # path = 'D:/JData_New/'
    path = '/home/peter/jdata-new/'
    os.chdir(path)

    APR_DATA = "JData_Action_201604.csv"
    USER_TABLE = "user_table.csv"
    ITEM_TABLE = "item_table.csv"
    TRAIN_DATA = "train.csv"

    user_table = pd.read_csv(USER_TABLE, header=0)
    item_table = pd.read_csv(ITEM_TABLE, header=0)

    apr_action = get_action_data(APR_DATA)

    behavior_today = behavior(apr_action, "2016-04-10", "2016-04-11", features_today)
    behavior_past = behavior(apr_action, "2016-04-05", "2016-04-10", features_past)
    behavior_dependent = behavior(apr_action, "2016-4-11", "2016-04-16", dependent)

    train_data = merge_data(behavior_past, behavior_today, behavior_dependent)

    train_data.to_csv(TRAIN_DATA, index=False)
