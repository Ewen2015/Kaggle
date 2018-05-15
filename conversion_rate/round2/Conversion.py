import time
import pandas as pd 
from BayesianSM import HyperParam

wd = '/Users/ewenwang/Documents/practice_data/conversion_rate/'
file = ['round1_ijcai_18_train_20180301.txt', 'round1_ijcai_18_test_a_20180301.txt', 'round1_ijcai_18_test_b_20180418.txt']

def timestamp_datetime(timestamp):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

def CVR(data, feature):
    """Calculate CVR features for categorical features."""
    
    data = data[data.is_trade.notnull()]

    feature_clk_df = pd.DataFrame(data.groupby([feature], as_index=False).size().reset_index())
    feature_clk_df.rename(columns={0:feature+'_count'}, inplace=True)

    feature_cv_df = pd.DataFrame(data[(data.is_trade==1)].groupby([feature], as_index=False).size().reset_index())
    feature_cv_df.rename(columns={0:feature+'_trade_count'}, inplace=True)

    feature_cvr_df = feature_clk_df.merge(feature_cv_df, on=feature, how='left')

    hyper = HyperParam(1, 1)
    I, C = feature_cvr_df[feature+'_count'], feature_cvr_df[feature+'_trade_count']
    hyper.update_from_data_by_moment(I, C)

    feature_cvr_df[feature+'_cvr'] = (C + hyper.alpha) / (I + hyper.alpha + hyper.beta)

    return feature_cvr_df

def RunCVR_slider(data, days_ahead=1):
    id_list = ['user_id', 'item_id', 'shop_id', 'context_id']
    date_list = range(18, 26-days_ahead)
    new_data = pd.DataFrame()
    for date in date_list:
        date_df = data[(data.day==date+days_ahead)]
        for feature in id_list:
            df_add = CVR(data=data[(data.day >= date) & (data.day < date+days_ahead)], feature=feature)
            date_df = date_df.merge(df_add, on=feature, how='left')   
        if date==18:
            new_data = date_df
        else:
            new_data = pd.concat([new_data, date_df])
    return new_data

# ==========

def PreCVR(data, feature):
    feature_clk_df = pd.DataFrame(data.groupby([feature], as_index=False).size().reset_index()).fillna(0)
    feature_clk_df.rename(columns={0:feature+'_count'}, inplace=True)

    feature_cv_df = pd.DataFrame(data[(data.is_trade==1)].groupby([feature], as_index=False).size().reset_index()).fillna(0)
    feature_cv_df.rename(columns={0:feature+'_trade_count'}, inplace=True)

    feature_cvr_df = feature_clk_df.merge(feature_cv_df, on=feature, how='left').fillna(0)
    return feature_cvr_df

def BayesainSM(I, C):
    hyper = HyperParam(1, 1)
    hyper.update_from_data_by_moment(I, C)
    return (hyper.alpha, hyper.beta)

def RunBayesian(data):
    id_list = ['user_id', 'item_id']

    data_train = data[data.is_trade.notnull()]
                      
    for feature in id_list:
        cvr_df = pd.DataFrame()
        cvr_df = PreCVR(data_train, feature)
        alpha, beta = BayesainSM(cvr_df[feature+'_count'], cvr_df[feature+'_trade_count'])
        
        new_data = pd.DataFrame()
        
        for date in range(18, 26):
            date_df = data[(data.day==date)]
            cvr_tem = PreCVR(data[data.day < date], feature)
            cvr_tem[feature+'_cvr'] = (cvr_tem[feature+'_trade_count'] + alpha) / (cvr_tem[feature+'_count'] + alpha + beta)
            date_df = date_df.merge(cvr_tem, on=feature, how='left') 
            if date==18:
                new_data = date_df
            else:
                new_data = pd.concat([date_df, new_data])
        
        data = data.merge(new_data[[feature+'_cvr', 'instance_id']], on='instance_id', how='left')

    drop_list = ['is_trade', 'user_id', 'item_id', 'context_timestamp', 'realtime', 'day']
    feature_cvr = [x for x in data.columns if x not in drop_list]

    cvr_data = data[feature_cvr]

    return cvr_data


if __name__ == '__main__':
    print('loading...')
    train, test_a, test_b = pd.read_csv(wd+file[0], sep=' '), pd.read_csv(wd+file[1], sep=' '), pd.read_csv(wd+file[2], sep=' ')
    data = pd.concat([train, test_a, test_b])
    features = ['instance_id', 'is_trade', 'user_id', 'item_id', 'context_timestamp']
    data = data[features]
    data['realtime'] = pd.to_datetime(data['context_timestamp'].apply(timestamp_datetime))
    data['day'] = data['realtime'].dt.day

    print('feature engineering...')
    cvr_data = RunBayesian(data)

    print('saving...')
    cvr_data.to_csv(wd+'cvr_bayesianSM_union.txt', index=False, sep=' ')