import LRS_SA_RGSS as LSR
import time
import pandas as pd
import xgboost as xgb
import lightgbm as lgbm
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')

def modelscore(y_test, y_pred):
    return log_loss(y_test, y_pred)

def add(x,y):
    return x + y

def substract(x,y):
    return x - y

def times(x,y):
    return x * y

def divide(x,y):
    return (x + 0.001)/(y + 0.001)


def main(temp, clf, CrossMethod, RecordFolder, test = False):
    # set up the data set first
    wd = '/mnt/resource/tm/'
    file = 'tmp-train.txt'

    print('loading...')
    t0=time.time()
    df = pd.read_csv(wd+file, sep='\t')
    df = df[~pd.isnull(df.is_trade)]
    print('\ttime spend: ', time.time()-t0)

    # get the features for selection
    drop_list = ['is_trade', 'instance_id', 'user_id', 'item_id', 'context_id', 'context_page_id', 'shop_id', 
        'day', 'hour', 'context_timestamp', 'wt_user_id', 'wt_user_occupation_id', 'wt_user_age_level', 'wt_user_star_level', 'wt_user_gender_id', 
        'item_pv_level_user_star_level_wt', 'item_pv_level_user_age_level_wt', 'item_pv_level_user_gender_id_wt',
        'item_city_id_user_star_level_wt',  'item_city_id_user_occupation_id_wt', 'item_city_id_user_gender_id_wt', 
        'user_star_level_item_brand_id_wt', 'user_star_level_match_cat_ct_wt', 'user_star_level_user_occupation_id_wt', 
        'user_star_level_item_collected_level_wt', 'user_star_level_item_sales_level_wt', 'user_star_level_context_page_id_wt', 
        'user_star_level_match_prop_ct_wt', 'user_star_level_shop_id_wt', 'user_star_level_user_age_level_wt', 
        'user_star_level_shop_star_level_wt', 'user_star_level_shop_review_num_level_wt', 'user_star_level_item_category_list_wt', 
        'user_star_level_user_gender_id_wt', 'user_star_level_item_price_level_wt',  'item_brand_id_user_gender_id_wt',
        'match_cat_ct_user_occupation_id_wt', 'match_cat_ct_user_gender_id_wt',  'user_occupation_id_item_collected_level_wt', 
        'user_occupation_id_item_sales_level_wt', 'user_occupation_id_context_page_id_wt', 'user_occupation_id_match_prop_ct_wt', 
        'user_occupation_id_shop_id_wt', 'user_occupation_id_user_age_level_wt', 'user_occupation_id_shop_star_level_wt', 
        'user_occupation_id_shop_review_num_level_wt', 'user_occupation_id_item_category_list_wt', 
        'user_occupation_id_user_gender_id_wt', 'user_occupation_id_item_price_level_wt', 'item_collected_level_user_age_level_wt',
        'item_collected_level_user_gender_id_wt', 'item_sales_level_user_age_level_wt', 'context_page_id_user_gender_id_wt',
        'match_prop_ct_user_age_level_wt', 'match_prop_ct_user_gender_id_wt', 'shop_id_user_age_level_wt',
        'shop_id_user_gender_id_wt', 'user_age_level_shop_star_level_wt', 'user_age_level_shop_review_num_level_wt', 
        'user_age_level_item_category_list_wt', 'user_age_level_user_gender_id_wt', 'user_age_level_item_price_level_wt',
        'shop_star_level_user_gender_id_wt', 'item_category_list_user_gender_id_wt', 'user_gender_id_item_price_level_wt',
        'shop_review_num_level_user_gender_id_wt', 'item_pv_level_user_occupation_id_wt', 'item_city_id_user_age_level_wt',
        'item_brand_id_user_occupation_id_wt', 'item_brand_id_user_age_level_wt', 'match_cat_ct_user_age_level_wt', 
        'item_sales_level_user_gender_id_wt', 'context_page_id_user_age_level_wt', 'shop_review_num_level_user_gender_id_wt'
        ]
    ColumnName = [x for x in df.columns if x not in drop_list] # + addcol #obtain columns withouth the useless features
    print(ColumnName)
    # start selecting
    a = LSR.LRS_SA_RGSS_combination(df=df,
                                    clf=clf,
                                    RecordFolder=RecordFolder,
                                    LossFunction=modelscore,
                                    label='is_trade',
                                    columnname=ColumnName[1::2], # the pattern for selection
                                    start=temp,
                                    CrossMethod=CrossMethod, # your cross term method
                                    PotentialAdd=[] # potential feature for Simulated Annealing
                                    )
    try:
        a.run()
    finally:
        with open(RecordFolder, 'a') as f:
            f.write('\n{}\n%{}%\n'.format(type,'-'*60))

if __name__ == "__main__":
    model = {'xgb': xgb.XGBClassifier(seed=1, max_depth=5, n_estimators=2000, nthread=-1),
             'lgb': lgbm.LGBMClassifier(random_state=1,num_leaves=29, n_estimators=1000),
             'lgb2': lgbm.LGBMClassifier(random_state=1,num_leaves=29, max_depth=5, n_estimators=1000),
             'lgb3': lgbm.LGBMClassifier(random_state=1, num_leaves=6, n_estimators=1000,max_depth=3,learning_rate=0.09, n_jobs=-1),
             'lgb4': lgbm.LGBMClassifier(random_state=1, num_leaves=6, n_estimators=5000,max_depth=3,learning_rate=0.095, n_jobs=-1),
             'lgb5': lgbm.LGBMClassifier(random_state=1, num_leaves=13, n_estimators=5000,max_depth=4,learning_rate=0.05, n_jobs=-1),
             'lgb6': lgbm.LGBMClassifier(random_state=1, num_leaves=6, n_estimators=5000,max_depth=3,learning_rate=0.05, n_jobs=-1)
            } # algorithm group

    CrossMethod = {'+':add,
                   '-':substract,
                   '*':times,
                   '/':divide,}

    RecordFolder = 'record.log' # result record file
    modelselect = 'lgb6' # selected algorithm

    temp = [
    'shop_score_description', 'user_gender_id', 'user_age_level', 'user_star_level', 'item_city_id', 'item_price_level', 'item_collected_level', 'item_pv_level', 'shop_star_level', 'wt_item_id', 'wt_item_category_list', 'match_prop_ct_shop_id_wt', 'item_city_id_shop_id_wt', 'item_city_id_context_page_id_wt', 'list_wt_item_property_list', 'wt_item_city_id', 'match_cat_ct_shop_id_wt', 'context_page_id_item_category_list_wt', 'wt_item_sales_level'    ] # start features combination
    main(temp,model[modelselect], CrossMethod, RecordFolder, test=False)
