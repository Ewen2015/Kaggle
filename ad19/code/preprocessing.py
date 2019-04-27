import time
import json
import pandas as pd 

ts_col = ['sam_id', 'ad_id', 'est_time', 'ad_size', 'ad_ind_id', 'item_type', 'item_id', 'ad_acc_id', 'period', 'user', 'price']

usr_col = ['user_age', 'user_gender', 'user_area', 'user_education', 'user_connectionType', 'user_behavior']


def timestamp_datetime(timestamp):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

def extend_time(df, time_col):
    df['realtime'] = df[time_col].apply(timestamp_datetime)
    df['realtime'] = pd.to_datetime(df['realtime'])
    df[time_col+'_year'] = df['realtime'].dt.year
    df[time_col+'_mon'] = df['realtime'].dt.month
    df[time_col+'_day'] = df['realtime'].dt.day
    df[time_col+'_weekday'] = df['realtime'].dt.dayofweek
    df[time_col+'_hour'] = df['realtime'].dt.hour
    del df[time_col]
    return df


def convert_period(code):
    binary = bin(code)

    start = 0
    duration = 0
    end = 0

    for b in [int(i) for i in str(binary[:1:-1])]:
        if b == 0:
            start += 0.5
        else:
            duration += 0.5
    end = start + duration

    return start, end, duration

def extend_period(df, col):
    df[col+'_start'], df[col+'_end'], df[col+'_dur'] = zip(*df[col].apply(lambda x: convert_period(x)))
    del df[col]
    return df

def extend_all_periods(df):
    for i in range(7):
        df['period_'+str(i)] = df['period'].apply(lambda x: int(str(x).split(',')[i]))
        df = extend_period(df, 'period_'+str(i))
    return df



def user_attr_cnt(df):
    df['user_attr_cnt'] = df['user'].apply(lambda x: len(str(x).split('|')))
    return df

def user_dict(string):
    coll = dict()
    if string == 'all':
        return coll
    for sec in string.split('|'):
        coll['user_'+sec.split(':')[0]] = sec.split(':')[1]
    return coll

def extend_user(df):
    df['user_dict'] = df['user'].apply(user_dict)

    df = pd.concat([df, pd.DataFrame(columns=['user_age', 'user_gender', 'user_area', 'user_education', 
                                              'user_connectionType', 'user_behavior', 'user_device',
                                              'user_consuptionAbility', 'user_status', 'user_work'])])

    for ind, dic in enumerate(['user_dict']):
        for c in [*dic.keys()]:
            df[c][ind] = dic[c]
    del df['user']
    del df['user_dict']
    return df 


def preprocessing(df):

    df = (df.pipe(extend_time, 'est_time')
            .pipe(extend_all_periods)
            .pipe(user_attr_cnt)
            .pipe(extend_user))

    return df 


def main():
    config = dict()

    with open('config.json', 'r') as f:
        config = json.load(f)

    test = pd.read_csv(config['dir_data']+config['file_ts_sm'], sep='\t', names=ts_col)

    test = preprocessing(test)


if __name__ == '__main__':
    main()