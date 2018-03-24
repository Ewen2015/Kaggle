import pandas as pd
import time

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

def pre_process(data):

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

category_features = ['item_id', 'item_brand_id', 'item_city_id', 
'user_id', 'user_gender_id', 'user_occupation_id', 
'context_id', 'context_page_id', 
'shop_id', 
'category_1', 'category_2', 'property_0', 'property_1', 'property_2', 
'predict_category_0', 'predict_category_1', 'predict_category_2',]

def commom(train, test, category_features):
    results = pd.DataFrame()
    for ind, val in enumerate(category_features):
        results[ind, 'column'] = val
        results[ind,'count_test'] = len(test[val].unique())
        results[ind,'count_train'] = len(train[val].unique())
        results[ind,'count_common'] = pd.merge(test, train, how='inner', on=[val]).groupby(val).size().shape[0]
        results[ind,'pct_test'] = results[ind,'count_common']/results[ind,'count_test']
        results[ind,'pct_train'] = results[ind,'count_common']/results[ind,'count_train']
    results.sort_values(['pct_test'], accending=False)
    print(results[ind,:])
    return results



if __name__ == '__main__':
    train, test = load_data(data_list=new_data)
    results = commom(train, test, category_features=category_features)









