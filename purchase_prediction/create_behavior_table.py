import pandas as pd
from collections import Counter
import os

path = 'D:/JData_New/'
os.chdir(path)

ACTION_201602_FILE = "JData_Action_201602.csv"
ACTION_201603_FILE = "JData_Action_201603.csv"
ACTION_201604_FILE = "JData_Action_201604.csv"

BEHAVIOR_FILE = "behavior_table.csv"

# apply type count

def add_type_count(group):
    behavior_type = group.type.astype(int)
    type_cnt = Counter(behavior_type)

    group['browse_num'] = type_cnt[1]
    group['addcart_num'] = type_cnt[2]
    group['delcart_num'] = type_cnt[3]
    group['buy_num'] = type_cnt[4]
    group['favor_num'] = type_cnt[5]
    group['click_num'] = type_cnt[6]

    return group[['user_sku', #'user_id', 'sku_id',
                  'browse_num', 'addcart_num', 'delcart_num',
                  'buy_num', 'favor_num', 'click_num']]

# def add_user_sku_column(fname):
#     df_us = pd.read_csv(fname, header=0)
#     df_us["user_sku"] = df_us['user_id'].map(int).map(str) + "_" + reader["sku_id"].map(str)
    
#     return df_us


def get_from_action_data(fname, chunk_size=100000):
    reader = pd.read_csv(fname, header=0, iterator=True)
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)[["user_sku", "type"]]
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped")

    df_ac = pd.concat(chunks, ignore_index=True)

    df_ac = df_ac.groupby(['user_sku'], as_index=False).apply(add_type_count)
    # Select unique row
    df_ac = df_ac.drop_duplicates('user_sku')

    return df_ac


def merge_action_data():
    df_ac = []
    df_ac.append(get_from_action_data(fname=ACTION_201602_FILE))
    df_ac.append(get_from_action_data(fname=ACTION_201603_FILE))
    df_ac.append(get_from_action_data(fname=ACTION_201604_FILE))

    df_ac = pd.concat(df_ac, ignore_index=True)
    df_ac = df_ac.groupby(['user_sku'], as_index=False).sum()

    return df_ac


if __name__ == "__main__":

    user_behavior = merge_action_data()

    user_behavior.to_csv(BEHAVIOR_FILE, index=False)
