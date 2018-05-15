import pandas as pd 
import lightgbm as lgb
from sklearn.metrics import log_loss
from gossipcat import SimulatedAnneal
import warnings

warnings.filterwarnings('ignore')

wd = '/Users/ewenwang/Documents/practice_data/conversion_rate/'
test_file = ['round1_ijcai_18_test_a_20180301.txt', 'round1_ijcai_18_test_b_20180418.txt']

file = 'snow_data.txt'
file_slider3 = 'snow_data_slider3.txt'
file_UISC = 'snow_data_UISC.txt'
file_UISC_union = 'snow_data_UISC_union.txt'
file_cap = 'cap_data.txt'

feature_cvr = 'feature_cvr.txt'
feature_conversion = 'feature_conversion.txt'
cvr_sm = 'cvr_bayesianSM.txt'
cvr_sm_union = 'cvr_bayesianSM_union.txt'

pagerank='pagerank.txt'
pagerank_union = 'pagerank_union.txt'

output_file = 'result.txt'

params = {'boosting_type': 'gbdt',
          'max_depth' : 2,
          'objective': 'binary', 
          'num_leaves': 4, 
          'learning_rate': 0.01, 
          'subsample': 0.6, 
          'colsample_bytree': 0.8, 
          'metric' : 'binary_logloss'}

gridParams = {
    'max_depth': [i for i in range(2, 17, 2)],
    'num_leaves': [i for i in range(20, 81, 4)],
    'colsample_bytree' : [i / 100.0 for i in range(50, 101, 5)],
    'subsample' : [i / 100.0 for i in range(50, 101, 5)]
    }

target = 'is_trade'
drop_list = ['is_trade']

def Submission(data, submit=False):

    params = {'boosting_type': 'gbdt',
            'max_depth' : 3,
            'objective': 'binary', 
            'num_leaves': 8, 
            'learning_rate': 0.01, 
            'subsample': 0.6, 
            'colsample_bytree': 0.8, 
            'metric' : 'binary_logloss'}

    gridParams = {
      'max_depth': [i for i in range(3, 8, 1)],
      'colsample_bytree' : [i / 10.0 for i in range(3, 11, 1)],
      'subsample' : [i / 10.0 for i in range(6, 11, 1)]
    }
    target = 'is_trade'
    drop_list = ['is_trade', 'item_category_list', 'item_property_list', 'predict_category_property', 'realtime', 'context_timestamp']
    features = [x for x in data.columns if x not in drop_list]

    if submit:
        train = data[data.is_trade.notnull()]
        test = data[data.is_trade.isnull()]
    else:
        test_day = 24
        train = data[(data['day'] >= 18) & (data['day'] < test_day)]
        test = data[(data['day'] == test_day)]

    mdl = lgb.LGBMClassifier(boosting_type= 'gbdt', 
              objective = 'binary', 
              n_jobs = -1, 
              silent = True,
              random_state = 0,
              max_depth = params['max_depth'],
              colsample_bytree = params['colsample_bytree'],
              subsample = params['subsample'])

    sim = SimulatedAnneal(mdl, gridParams,  scoring='neg_log_loss', alpha=0.75, n_trans=10, cv=4, verbose=True, random_state=0)

    print('simulated annealing...')
    sim.fit(train[features], train[target])

    # Using parameters already set above, replace in the best from the simulated annealing search
    params['max_depth'] = sim.best_params_['max_depth']
    params['colsample_bytree'] = sim.best_params_['colsample_bytree']
    params['subsample'] = sim.best_params_['subsample']

    print('\nbest score:', '{:.6f}'.format(sim.best_score_), 
          '\nbest parameters:', str({key: '{:.3f}'.format(value) for key, value in sim.best_params_.items()}))

    print('predicting...')
    test['predicted_score'] = sim.best_estimator_.predict_proba(test[features])[:,1]
    
    if submit:
        result = test[['instance_id', 'predicted_score']]
        result = pd.DataFrame(pd.read_csv(wd+test_file[1], sep=' ')['instance_id']).merge(result, on='instance_id', how='left').fillna(0)
        print('\nsaving...')
        result.to_csv(wd+output_file, sep=' ', index=False)
    else:
        print('Log loss: ', log_loss(test['is_trade'], test['predicted_score']))   
    
    return None

def SubmissionCross(data, submit=False):

    features = [x for x in data.columns if x not in drop_list]

    if submit:
        test_day = 25
        train = data[data.is_trade.notnull()]
        test = data[data.is_trade.isnull()]
    else:
        test_day = 24
        train = data[(data['day'] >= 18) & (data['day'] < test_day)]
        test = data[(data['day'] == test_day)]

    lgb_train = lgb.Dataset(train[features], train[target], free_raw_data=False)

    mdl = lgb.LGBMClassifier(boosting_type= 'gbdt', 
              objective = 'binary', 
              n_jobs = -1, 
              silent = True,
              random_state = 0,
              max_depth = params['max_depth'],
              max_bin = params['max_bin'], 
              subsample_for_bin = params['subsample_for_bin'],
              subsample = params['subsample'], 
              subsample_freq = params['subsample_freq'], 
              min_split_gain = params['min_split_gain'], 
              min_child_weight = params['min_child_weight'], 
              min_child_samples = params['min_child_samples'], 
              scale_pos_weight = params['scale_pos_weight'])

    # Create the grid
    # grid = GridSearchCV(mdl, gridParams, verbose=1, cv=4, n_jobs=-1)
    sim = SimulatedAnneal(mdl, gridParams,  scoring='neg_log_loss', alpha=0.75, n_trans=5, cv=4, verbose=True, random_state=0)

    print('simulated annealing...')
    # Run the grid
    sim.fit(train[features], train[target])

    # Using parameters already set above, replace in the best from the simulated annealing search
    params['max_depth'] = sim.best_params_['max_depth']
    params['num_leaves'] = sim.best_params_['num_leaves']
    params['colsample_bytree'] = sim.best_params_['colsample_bytree']
    params['subsample'] = sim.best_params_['subsample']

    print('\nbest score:', '{:.6f}'.format(sim.best_score_), 
          '\nbest parameters:', str({key: '{:.3f}'.format(value) for key, value in sim.best_params_.items()}))

    # Train  
    print('\ntraining...')   
    gbm = lgb.train(params, 
                    lgb_train, 
                    num_boost_round=100000, 
                    valid_sets=lgb_train,
                    early_stopping_rounds = 500,
                    verbose_eval=100)

    print('\npredicting...')
    test['predicted_score'] = gbm.predict(test[features], num_iteration=gbm.best_iteration)
    
    if submit:
        result = test[['instance_id', 'predicted_score']]
        result = pd.DataFrame(pd.read_csv(wd+test_file[2], sep=' ')['instance_id']).merge(result, on='instance_id', how='left').fillna(0)
        print('\nsaving...')
        result.to_csv(wd+output_file, sep=' ', index=False)
    else:
        print('Log loss: ', log_loss(test['is_trade'], test['predicted_score']))   
    
    return None

def SubmissionSlide(data, validation_length=2):

    params = {'boosting_type': 'gbdt',
          'max_depth' : 2,
          'objective': 'binary', 
          'num_leaves': 4, 
          'learning_rate': 0.01, 
          'subsample': 0.6, 
          'colsample_bytree': 0.8, 
          'metric' : 'binary_logloss'}

    features = [x for x in data.columns if x not in drop_list]

    train = data[data.is_trade.notnull()]
    test = data[data.is_trade.isnull()]

    gbm = None

    print('\ntraining...')
    for i, date in enumerate(range(18+validation_length, 25), 1):
        print('\ntraining round: ', str(i), '\tvalidation date: ', str(date))
        train_temp = train[(train['day'] >= date-validation_length) & (train['day'] < date)]
        valid = train[(train['day'] == date)]

        lgb_train = lgb.Dataset(train_temp[features], train_temp[target], free_raw_data=False)
        lgb_valid = lgb.Dataset(valid[features], valid[target], reference=lgb_train, free_raw_data=False)

        gbm = lgb.train(params=params, 
                        train_set=lgb_train,
                        valid_sets=lgb_valid,
                        init_model=None if date==18+validation_length else gbm, 
                        keep_training_booster=True,
                        num_boost_round=10000, 
                        early_stopping_rounds = 500,
                        verbose_eval=100)

    print('\npredicting...')
    test['predicted_score'] = gbm.predict(test[features], num_iteration=gbm.best_iteration)

    result = test[['instance_id', 'predicted_score']]
    result = pd.DataFrame(pd.read_csv(wd+test_file[2], sep=' ')['instance_id']).merge(result, on='instance_id', how='left').fillna(0)
    
    print('\nsaving...')
    result.to_csv(wd+output_file, sep=' ', index=False)


def SubmissionSimple(data, submit=False):
  features = [x for x in data.columns if x not in drop_list]

  train = data[data.is_trade.notnull()]
  test = data[data.is_trade.isnull()]

  lgb_train = train[(train['day'] >= 18) & (train['day'] < 24)]
  lgb_valid = train[(train['day'] == 24)]

  lgb_train = lgb.Dataset(lgb_train[features], lgb_train[target], free_raw_data=False)
  lgb_valid = lgb.Dataset(lgb_valid[features], lgb_valid[target], reference=lgb_train, free_raw_data=False)

  print('\ntraining...')
  gbm = lgb.train(params=params, 
                  train_set=lgb_train,
                  valid_sets=[lgb_train, lgb_valid],
                  num_boost_round=10000, 
                  early_stopping_rounds=500,
                  verbose_eval=100)

  if submit:
    print('\npredicting...')
    test['predicted_score'] = gbm.predict(test[features], num_iteration=gbm.best_iteration)
    result = test[['instance_id', 'predicted_score']]
    result = pd.DataFrame(pd.read_csv(wd+test_file[2], sep=' ')['instance_id']).merge(result, on='instance_id', how='left').fillna(0)
    print('\nsaving...')
    result.to_csv(wd+output_file, sep=' ', index=False)
    
  return gbm



if __name__ == '__main__':
    print('loading...')
    data = pd.read_csv(wd+file_UISC_union, sep=' ')
    data = data.merge(pd.read_csv(wd+pagerank_union, sep=' '), on='instance_id', how='left').merge(pd.read_csv(wd+cvr_sm_union, sep=' '), on='instance_id', how='left')
    # data = data.merge(pd.read_csv(wd+pagerank_union, sep=' '), on='instance_id', how='left')

    # SubmissionSlide(data, validation_length=6)
    # SubmissionSimple(data, submit=True)
    # Submission(data, submit=False)  # bayesian smoothing: update_from_data_by_moment
    Submission(data, submit=True)
    