import pandas as pd
import numpy as np
import time

def timestamp_datetime(timestamp):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

def commom(train, test, categorical_features):
    results = pd.DataFrame(columns=['column', 'count_test', 'count_train', 'count_common', 'pct_test', 'pct_train'])
    for ind, val in enumerate(categorical_features):
        df = pd.DataFrame()
        df.loc[ind, 'column']  = val
        df.loc[ind, 'count_test'] = len(test[val].unique())
        df.loc[ind, 'count_train']  = len(train[val].unique())
        df.loc[ind, 'count_common']  = pd.merge(test, train, how='inner', on=[val]).groupby(val).size().shape[0]
        df.loc[ind, 'pct_test'] = df.loc[ind, 'count_common']/df.loc[ind, 'count_test']
        df.loc[ind, 'pct_train'] = df.loc[ind, 'count_common']/df.loc[ind, 'count_train']
        results = results.append(df)
    results.sort_values(by=['pct_test'], ascending=False)
    
    return results


class Data(object):
    """docstring for Data"""
    def __init__(self, wd, data_list, is_all=True):
        super(Data, self).__init__()

        if is_all:
            self.all_data = pd.read_csv(wd+data_list[0], sep=' ')
        else:
            self.train = pd.read_csv(wd+data_list[0], sep=' ')
            self.test = pd.read_csv(wd+data_list[1], sep=' ')
            self.all_data = self.train.append(self.test)

        self.dtrain = pd.DataFrame()
        self.dvalidation = pd.DataFrame()
        self.dall = pd.DataFrame()

        self.target = 'is_trade'
        self.drop_list = ['instance_id', 'time', 'day']
        self.features = ['item_id', 'item_brand_id', 'item_city_id', 
        'user_id', 'user_gender_id', 'user_occupation_id', 
        'context_id', 'context_page_id', 
        'shop_id', 
        'category_1', 'category_2', 'property_0', 'property_1', 'property_2', 
        'predict_category_0', 'predict_category_1', 'predict_category_2',
        'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 
        'user_age_level', 'user_star_level', 
        'shop_score_service', 'shop_score_delivery', 'shop_score_description', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 
        'week', 'hour', 'user_query_day', 'user_query_day_hour']
        self.id_features = ['item_id', 'item_brand_id', 'item_city_id', 
        'user_id', 'user_gender_id', 'user_occupation_id', 
        'context_id', 'context_page_id', 
        'shop_id']
        self.numeric_features = ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 
        'user_age_level', 'user_star_level', 
        'shop_score_service', 'shop_score_delivery', 'shop_score_description', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 
        'week', 'hour', 'user_query_day', 'user_query_day_hour']
        self.categorical_features = ['item_id', 'item_brand_id', 'item_city_id', 
        'user_id', 'user_gender_id', 'user_occupation_id', 
        'context_id', 'context_page_id', 
        'shop_id', 
        'category_1', 'category_2', 'property_0', 'property_1', 'property_2', 
        'predict_category_0', 'predict_category_1', 'predict_category_2']
        self.predictors = []
        
    def Save(self, data, file):
        data.to_csv(wd+file, index=False, sep=' ')
        print('Saved.')
        return None

    def Preprocess(self, data):
        """Preprocess the raw data."""
        for i in range(3):
            data['category_%d'%(i)] = data['item_category_list'].apply(lambda x: x.split(";")[i] if len(x.split(";"))>i else " ")
        del data['item_category_list']
        del data['category_0']

        for i in range(3):
            data['property_%d'%(i)] = data['item_property_list'].apply(lambda x: x.split(";")[i] if len(x.split(";"))>i else " ")
        del data['item_property_list']

        for i in range(3):
            data['predict_category_%d'%(i)] = data['predict_category_property'].apply(lambda x: str(x.split(";")[i]).split(":")[0] if len(x.split(";"))>i else " ")
        del data['predict_category_property']

        data['time'] = data.context_timestamp.apply(timestamp_datetime)
        del data['context_timestamp']
        data['time_tmp'] = pd.to_datetime(data['time'])
        data['day'] = data['time_tmp'].dt.day
        data['week'] = data['time_tmp'].dt.weekday
        data['hour'] = data['time_tmp'].dt.hour
        del data['time_tmp']
        user_query_day = data.groupby(['user_id', 'day']).size().reset_index().rename(columns={0: 'user_query_day'})
        data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
        user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(columns={0: 'user_query_day_hour'})
        data = pd.merge(data, user_query_day_hour, 'left',on=['user_id', 'day', 'hour'])

        return data

    def Validation(self, all_data, validation_day=[24]):                                # array([18, 21, 19, 20, 22, 23, 24, 25])
        """Generate training and validation sets."""
        self.dtrain = all_data.loc[all_data.day < np.array(validation_day).min()]
        self.dvalidation = all_data.loc[all_data.day.isin(validation_day)]
        
        return self.dtrain, self.dvalidation

    def EngCVR(self, train, feature):
        """Calculate CVR features for categorical features."""
        feature_clk_df = pd.DataFrame(train.groupby([feature], as_index=False).size().reset_index())
        feature_clk_df.rename(columns={0:feature+'_count'}, inplace=True)

        feature_cv_df = pd.DataFrame(train[(train.is_trade==1)].groupby([feature], as_index=False).size().reset_index())
        feature_cv_df.rename(columns={0:feature+'_trade_count'}, inplace=True)

        feature_cvr_df = feature_clk_df.merge(feature_cv_df, on=feature, how='left')
        feature_cvr_df[feature+'_cvr'] = feature_cvr_df[feature+'_trade_count']/feature_cvr_df[feature+'_count']

        return feature_cvr_df

    def ValidationFE(self, validation_day=[24]):                                # array([18, 21, 19, 20, 22, 23, 24, 25])
        """Generate training and validation sets and do feature enginerring."""
        self.dtrain, self.dvalidation = self.Validation(all_data = self.all_data, validation_day=validation_day)
        self.dall = self.dtrain.append(self.dvalidation)    # used for trainging and validation, not always equal to all_data

        for feature in self.categorical_features:
            new_feature = self.EngCVR(train=self.dtrain, feature=feature)
            self.dall = self.dall.merge(new_feature, on=feature, how='left')

        self.predictors = [x for x in self.dall.columns if x not in [self.target]+self.drop_list+self.categorical_features]
        self.dtrain, self.dvalidation = self.Validation(all_data = self.dall, validation_day=validation_day)

        return self.dtrain, self.dvalidation



if __name__ == '__main__':
    pass







