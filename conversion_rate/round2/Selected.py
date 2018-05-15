import pandas as pd 
import warnings
warnings.filterwarnings('ignore')

wd = '/mnt/resource/tm/'
base = ['tmp-train.txt', 'tmp-round2.txt']
out_file = ['round2_peter_train.txt', 'round2_peter_test.txt']

features = ['instance_id', 
'wt_item_id', 'wt_item_category_list', 'match_prop_ct_shop_id_wt', 
'item_city_id_shop_id_wt', 'item_city_id_context_page_id_wt', 'list_wt_item_property_list', 'wt_item_city_id', 
'match_cat_ct_shop_id_wt', 'context_page_id_item_category_list_wt', 'item_brand_id_match_prop_ct_wt', 
'context_page_id_shop_star_level_wt'
]

print('loading...')
print('/ttrain')
train = pd.read_csv(wd+base[0], sep='\t')
train = train[features]
print('/t/tsaving...')
train.to_csv(wd+out_file[0], index=False, sep=' ')
print('done')

del train

print('test')
test = pd.read_csv(wd+base[1], sep='\t')
test = test[features]
print('/t/tsaving...')
test.to_csv(wd+out_file[1], index=False, sep=' ')

print('done')