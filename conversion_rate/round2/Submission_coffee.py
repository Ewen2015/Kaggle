import time
import pandas as pd 
import lightgbm as lgb
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')

wd = ['/mnt/resource/tm/', '/mnt/resource/tm/code/']

file = ['train-9.txt', 'valid-10.txt', 'test-11.txt']
record_file = 'record.log'

def Submission(drop_list=None):

    print('loading...')
    train = pd.read_csv(wd[0]+file[0], sep='\t')
    valid = pd.read_csv(wd[0]+file[1], sep='\t')

    target = 'is_trade'
    if drop_list == None:
        drop_list = ['is_trade', 'instance_id', 'user_id', 'item_id', 'context_id', 'context_page_id', 'shop_id', 
        'day', 'hour', 'context_timestamp']

    features = [x for x in train.columns if x not in drop_list]
    print('features:\n', features)

    X = train[features]
    y = train[target].values
    X_val = valid[features]
    y_val = valid[target].values

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
    lgb_model_1 = lgb_1.fit(X, y, eval_set=[(X_val, y_val)], early_stopping_rounds=200, verbose=False, callbacks=[lgb.print_evaluation(100)])
    print('\ttime spend: ', time.time()-t0)

    best_iter = lgb_model_1.best_iteration_
    best_score = lgb_model_1.best_score_

    print('best_iter: ', best_iter, '\nbest_score: ', best_score)

    with open(wd[1]+record_file, 'a') as f:
        f.write('best iteration:\t{0}\n\tbest score: \t{1}\n'.format(best_iter, best_score))

    del train
    del valid

    print('loading...')
    test = pd.read_csv(wd[0]+file[2], sep='\t')

    print('predicting...')
    t0=time.time()
    pred = lgb_model_1.predict_proba(test[features])[:, 1]
    print('\ttime spend: ', time.time()-t0)

    print('logloss for test: ', log_loss(test[target], pred))

    return lgb_model_1

if __name__ == '__main__':
    Submission()