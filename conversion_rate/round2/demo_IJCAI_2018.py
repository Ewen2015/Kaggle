import FeatureSelection as FS
from sklearn.metrics import log_loss
import lightgbm as lgbm
import pandas as pd
import time

wd = '/Users/ewenwang/Documents/practice_data/conversion_rate/'
train_file = 'eng_train.txt'

def timestamp_datetime(timestamp):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

# def prepareData():
#     """prepare you dataset here"""
#     print('loading...')
#     df = pd.read_csv(wd+train_file, sep=' ', na_values=[-1])
#     df = (df.merge(pd.read_csv(wd+match, sep=' '), on='instance_id', how='left')
#             .merge(pd.read_csv(wd+page, sep=' '), on='instance_id', how='left'))
#     df['realtime'] = df['context_timestamp'].apply(timestamp_datetime)
#     df['realtime'] = pd.to_datetime(df['realtime'])
#     df['hour'] = df['realtime'].dt.hour
#     df['minute'] = df['realtime'].dt.minute
#     df['minute_of_day'] = df['hour']*60+df['minute']
#     del df['realtime']
#     print('done.')
#     return df

def prepareData():
    """prepare you dataset here"""
    print('loading...')
    df = pd.read_csv(wd+train_file, sep=' ', na_values=[-1])
    print('done')
    return df


def modelscore(y_test, y_pred):
    """set up the evaluation score"""
    return log_loss(y_test, y_pred)

def validation(X,y,features, clf,lossfunction):
    """set up your validation method"""
    totaltest = 0
    for H in [11]:
        T = (X.hour != H)
        X_train, X_test = X[T], X[~T]
        X_train, X_test = X_train[features], X_test[features]
        y_train, y_test = y[T], y[~T]
        clf.fit(X_train,y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='logloss', verbose=False, early_stopping_rounds=200)
        totaltest += lossfunction(y_test, clf.predict_proba(X_test)[:,1])
    totaltest /= 1.0
    return totaltest

def add(x,y):
    return (x + y).round(2)

def substract(x,y):
    return (x - y).round(2)

def times(x,y):
    return (x * y).round(2)

def divide(x,y):
    return ((x + 0.001)/(y + 0.001)).round(2)

def deviation(x,y):
    avg = y.groupby(x).mean()
    sd = y.groupby(x).std()
    return pd.concat([x, y], axis=1).apply(lambda row: (row[1]-avg[row[0]]+0.001)/(sd[row[0]]+0.001), axis=1).round(2)

def ratio(x, y):
    cnt = x.groupby(x).size()
    tot = len(x)
    return x.apply(lambda row: (cnt[row]+0.001)/(tot+0.001)).round(2)

def probability(x, y):
    df = pd.concat([x, y], axis=1)
    x_cnt = x.groupby(x).size()
    xy_cnt = df.groupby(by=[x, y]).size()
    return df.apply(lambda row: (xy_cnt[row[0], row[1]]+0.001)/(x_cnt[row[0]]+0.001), axis=1).round(2)


CrossMethod = {'+':add,
               '-':substract,
               '*':times,
               '/':divide,
               # 'd':deviation,
               # 'r':ratio,
               # 'p':probability,
               }

def main():
    df = prepareData()
    drop_list = [
        'is_trade', 'instance_id',
        'item_category_list', 'item_property_list', 'predict_category_property', 
        'context_realtime', 'context_hour', 'context_day', 'realtime'
        ]
    features = ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']

    sf = FS.Select(Sequence = True, Random = True, Cross = False)    # select the way you want to process searching
    sf.ImportDF(df,label = 'is_trade')
    sf.ImportLossFunction(modelscore, direction = 'descend')
    sf.ImportCrossMethod(CrossMethod)
    sf.InitialNonTrainableFeatures(drop_list)
    sf.InitialFeatures(features)
    sf.GenerateCol(key = 'mean', selectstep = 2)                    # for getting rid of the useless columns in the dataset
    sf.SetSample(0.1, samplemode = 0, samplestate = 0)
    sf.SetFeaturesLimit(20)
    sf.SetTimeLimit(5)
    sf.clf = lgbm.LGBMClassifier(random_state=1, num_leaves = 6, n_estimators=5000, max_depth=3, learning_rate = 0.1, n_jobs=-1)
    sf.SetLogFile('recordml.log')
    sf.run(validation)

if __name__ == "__main__":
    main()
