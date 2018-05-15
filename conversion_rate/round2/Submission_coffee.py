import time
import pandas as pd 
import lightgbm as lgb
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')

wd = ['/mnt/resource/tm/', '/mnt/resource/tm/code/']

base = ['tmp-train.txt', 'tmp-round2.txt']
page = ['round2_page_train_7.txt', 'round2_page_test.txt']
union = ['round2_union_train_7.txt', 'round2_union_test.txt']

test_file = ['round2_ijcai_18_test_b_20180510.txt']
model_file = ['model_step_1.txt', 'model_step_2.txt']
record_file = 'record.log'

def Load(which_data, merge=False):

    if which_data=='train':
        print('loading...')
        t0=time.time()
        train = pd.read_csv(wd[0]+base[0], sep='\t')
        print('\ttime spend: ', time.time()-t0)

        if merge:
            print('training data merging...')
            t0=time.time()
            train = train.merge(pd.read_csv(wd[0]+page[0], sep=' '), on='instance_id', how='left')
            print('\ttime spend: ', time.time()-t0)

            print('train shape: ', train.shape)

            print('saving...')
            t0=time.time()
            train.to_csv(wd[0]+union[0], index=False, sep=' ')
            print('\ttime spend: ', time.time()-t0)
        return train 
    elif which_data=='test':
        print('loading...')
        t0=time.time()
        test = pd.read_csv(wd[0]+base[1], sep='\t')
        print('\ttime spend: ', time.time()-t0)

        if merge:
            print('test data merging...')
            t0=time.time()
            test = test.merge(pd.read_csv(wd[0]+page[1], sep=' '), on='instance_id', how='left')
            print('\ttime spend: ', time.time()-t0)

            print('test shape: ', test.shape)

            print('saving...')
            test.to_csv(wd[0]+union[1], index=False, sep=' ')
            print('\ttime spend: ', time.time()-t0)
        return test
    else:
        print('please choose which_data: train or test')
        return None

def Submission(drop_list=None, valid_hour=11, submit=False):

    train = Load(which_data='train')

    print('unique hours:\n', train.hour.unique())
    print('will validate on hour ', valid_hour, ' to get the best iteration number')
    if valid_hour>0:
        filter_ = (train.hour>=valid_hour)
        train_ = train[~filter_]
        valid_ = train[filter_]
    else:
        train_, valid_ = train_test_split(train, test_size=0.2, random_state=0)

    target = 'is_trade'
    if drop_list == None:
        drop_list = ['is_trade', 'instance_id', 'user_id', 'item_id', 'context_id', 'context_page_id', 'shop_id', 
        'day', 'hour', 'context_timestamp']

    features = [x for x in train.columns if x not in drop_list]
    print('features:\n', features)

    X = train_[features]
    y = train_[target].values
    X_tes = valid_[features]
    y_tes = valid_[target].values

    print('Training LGBM model...')
    t0=time.time()
    lgb_1 = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        num_leaves=16,
        depth=4,
        learning_rate=0.01,
        seed=2018,
        colsample_bytree=0.6,
        subsample=0.8,
        n_estimators=20000,
        silent = True)
    lgb_model_1 = lgb_1.fit(X, y, eval_set=[(X_tes, y_tes)], early_stopping_rounds=200, verbose=False, callbacks=[lgb.print_evaluation(100)])
    print('\ttime spend: ', time.time()-t0)

    best_iter = lgb_model_1.best_iteration_
    best_score = lgb_model_1.best_score_

    print('best_iter: ', best_iter, '\nbest_score: ', best_score)

    with open(wd[1]+record_file, 'a') as f:
        f.write('best iteration:\t{0}\n\tbest score: \t{1}\n'.format(best_iter, best_score))
    
    X_2 = train[features]
    y_2 = train[target].values

    print('Training LGBM model...')
    t0=time.time()
    lgb_2 = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        num_leaves=32,
        depth=4,
        learning_rate=0.01,
        seed=2018,
        colsample_bytree=0.6,
        subsample=0.9,
        n_estimators=best_iter,
        silent = True)

    lgb_model_2 = lgb_2.fit(X_2, y_2)
    print('\ttime spend: ', time.time()-t0)

    del train 

    # print('plotting...')
    # lgb.plot_importance(lgb_model_2, figsize=(12, 70))
    # plt.show()

    print('model saving...')
    lgb_model_1.save_model(wd[1]+model_file[0])

    if submit:
        test = Load(which_data='test')

        print('predicting...')
        t0=time.time()
        pred = lgb_model_2.predict_proba(test[features])[:, 1]
        print('\ttime spend: ', time.time()-t0)

        test['predicted_score'] = pred

        result = test[['instance_id', 'predicted_score']]
        result = pd.DataFrame(pd.read_csv(wd[0]+test_file[0], sep=' ')['instance_id']).merge(result, on='instance_id', how='left').fillna(0)
        
        print('\nresults saving...')
        t0=time.time()
        result.to_csv(wd[0]+'results.txt', sep=' ', index=False)
        print('\ttime spend: ', time.time()-t0)

    return lgb_model_2

if __name__ == '__main__':
    Submission(submit=True)