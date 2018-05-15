import time
import pandas as pd 
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 


def Merge(which_data):
    wd = ['/Users/ewenwang/Documents/practice_data/conversion_rate/', '/Users/ewenwang/Documents/GitHub/Kaggle/conversion_rate/round2/']

    uisc = ['round2_processed_train_7.txt', 'round2_processed_test.txt']
    page = ['round2_page_train_7.txt', 'round2_page_test.txt']
    peter = ['round2_peter_train.txt', 'round2_peter_test.txt']
    # union = ['round2_union_train_7.txt', 'round2_union_test.txt']

    if which_data=='train':
        print('loading...')
        t0=time.time()
        train = pd.read_csv(wd[0]+uisc[0], sep=' ')
        print('\ttime spend: ', time.time()-t0)

        print('training data merging...')
        t0=time.time()
        train = train.merge(pd.read_csv(wd[0]+page[0], sep=' '), on='instance_id', how='left')
        wt = pd.read_csv(wd[0]+peter[0], sep=' ')
        # wt = wt.drop('index')
        train = train.merge(wt, on='instance_id', how='left')
        del wt 
        print('\ttime spend: ', time.time()-t0)

        print('train shape: ', train.shape)

        # print('saving...')
        # t0=time.time()
        # train.to_csv(wd[0]+union[0], index=False, sep=' ')
        # print('\ttime spend: ', time.time()-t0)
        return train 
    elif which_data=='test':
        print('loading...')
        t0=time.time()
        test = pd.read_csv(wd[0]+uisc[1], sep=' ')
        print('\ttime spend: ', time.time()-t0)

        print('test data merging...')
        t0=time.time()
        test = test.merge(pd.read_csv(wd[0]+page[1], sep=' '), on='instance_id', how='left')
        wt = pd.read_csv(wd[0]+peter[1], sep=' ')
        # wt = wt.drop('index')
        test = test.merge(wt, on='instance_id', how='left')
        del wt 
        print('\ttime spend: ', time.time()-t0)

        print('test shape: ', test.shape)

        # print('saving...')
        # test.to_csv(wd[0]+union[1], index=False, sep=' ')
        # print('\ttime spend: ', time.time()-t0)
        return test
    else:
        print('please choose which_data: train or test')
        return None

def Submission(valid_hour=11):
    wd = ['/Users/ewenwang/Documents/practice_data/conversion_rate/', '/Users/ewenwang/Documents/GitHub/Kaggle/conversion_rate/round2/']

    test_file = ['round2_ijcai_18_test_b_20180510.txt']

    train = Merge(which_data='train')

    if valid_hour>0:
        filter_ = (train.hour>=valid_hour)
        train_ = train[~filter_]
        valid_ = train[filter_]
    else:
        train_, valid_ = train_test_split(train, test_size=0.2, random_state=0)

    target = 'is_trade'
    # if drop_list == None:
    #     drop_list = ['is_trade', 'instance_id', 'user_id', 'item_id', 'context_id', 'context_page_id', 'shop_id', 
    #     'hour', 'context_timestamp']

    # features = [x for x in train.columns if x not in drop_list]

    features = [
    'user_gender_id', 'user_age_level', 'user_star_level', 
    'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 
    'shop_star_level', 'shop_review_positive_rate', 'shop_score_service', 'shop_score_description', 
    
    'item_id_ratio', 'item_city_id_user_age_level_prob', 'item_collected_level_ratio', 'item_price_level_ratio',  
    'context_page_id_user_gender_id_prob', 'context_page_id_user_star_level_prob',
    'shop_score_service_bin_ratio', 'shop_star_level_ratio', 'shop_review_positive_rate_bin_ratio', 'shop_id_ratio', 'shop_review_num_level_ratio',
    'user_pagerank', 'hour_ratio',
    
    'wt_item_id', 'wt_item_category_list', 'match_prop_ct_shop_id_wt', 
    'item_city_id_shop_id_wt', 'item_city_id_context_page_id_wt', 'list_wt_item_property_list', 'wt_item_city_id', 
    'match_cat_ct_shop_id_wt', 'context_page_id_item_category_list_wt', 'item_brand_id_match_prop_ct_wt', 
    'context_page_id_shop_star_level_wt'
    ]

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
    test = Merge(which_data='test')

    print('predicting...')
    t0=time.time()
    pred = lgb_model_2.predict_proba(test[features])[:, 1]
    print('\ttime spend: ', time.time()-t0)

    test['predicted_score'] = pred

    result = test[['instance_id', 'predicted_score']]
    result = pd.DataFrame(pd.read_csv(wd[0]+test_file[0], sep=' ')['instance_id']).merge(result, on='instance_id', how='left').fillna(0)
    
    print('\nsaving...')
    t0=time.time()
    result.to_csv(wd[0]+'results.txt', sep=' ', index=False)
    print('\ttime spend: ', time.time()-t0)

    print('plotting...')
    lgb.plot_importance(lgb_model_2, figsize=(12, 25))
    plt.show()
    return lgb_model_2

if __name__ == '__main__':
    Submission()