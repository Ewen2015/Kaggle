import pandas as pd 
from lightgbm import LGBMClassifier

def Submission(train, test):

    target = 'is_trade'
    drop_list = ['is_trade', 'item_category_list', 'item_property_list', 'predict_category_property', 
    'instance_id', 'context_id', 'realtime', 'context_timestamp']
    features = [x for x in train.columns if x not in drop_list]

    gbm = LGBMClassifier(
        boosting_type='gbdt', 
        num_leaves=31, 
        max_depth=-1, 
        learning_rate=0.01,         #
        n_estimators=1100,          #
        subsample_for_bin=2000, 
        objective=None, 
        class_weight=None, 
        min_split_gain=0.0, 
        min_child_weight=0.001, 
        min_child_samples=20, 
        subsample=0.6,              #
        subsample_freq=5, 
        colsample_bytree=0.6,       #
        reg_alpha=0.0, 
        reg_lambda=0.0, 
        random_state=2018,          #
        n_jobs=-1, 
        silent=True
    )

    print('fitting...')
    gbm.fit(train[features], train[target])

    print('predicting...')
    test['predicted_score'] = gbm.predict_proba(test[features])[:, 1]
    result = test[['instance_id', 'predicted_score']]

    print('saving...')
    result.to_csv(wd+'result_3.txt', sep=' ', index=False)
    return None

if __name__ == '__main__':
    wd = '/Users/ewenwang/Documents/practice_data/conversion_rate/'
    file = 'snow_data.txt'

    data = pd.read_csv(wd+file, sep=' ')
    train = data[data.is_trade.notnull()]
    test = data[data.is_trade.isnull()]
    Submission(train, test)