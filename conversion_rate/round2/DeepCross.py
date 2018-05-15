import numpy as np
import pandas as pd
import time
import gc
import keras.backend as K
from keras import layers
from keras.layers import Dense, Input, Embedding, Reshape, Add, Flatten, merge, Lambda
from keras.models import Model
# from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')

def timestamp_datetime(timestamp):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

def encode_feature(values):
    return values.map(dict(zip(values.unique(), range(1, len(values.unique()) + 1))))

def process_data(data):
    print('preprocessing...')

    print('\tcategorical variables')
    data['item_id'] = encode_feature(data['item_id'])
    data['user_id'] = encode_feature(data['user_id'])
    data['context_id'] = encode_feature(data['context_id'])
    data['shop_id'] = encode_feature(data['shop_id'])

    data['item_brand_id'] = encode_feature(data['item_brand_id'])
    data['item_city_id'] = encode_feature(data['item_brand_id'])
    data['user_occupation_id'] = encode_feature(data['user_occupation_id'])
    data['context_page_id'] = encode_feature(data['context_page_id'])

    data['realtime'] = data['context_timestamp'].apply(timestamp_datetime)
    data['realtime'] = pd.to_datetime(data['realtime'])
    data['hour'] = data['realtime'].dt.hour
    del data['realtime']

    user_query_hour = data.groupby(['user_id', 'hour']).size().reset_index().rename(columns={0: 'user_query_hour'})
    data = pd.merge(data, user_query_hour, 'left', on=['user_id', 'hour'])

    print('\tcontinuous variables')

    label = data['is_trade']
    del data['is_trade']

    scaler = StandardScaler()
    cate_columns = ['item_id','user_id','context_id','shop_id', 'item_brand_id','item_city_id',
                    'user_gender_id','user_occupation_id','context_page_id'] 
    cont_columns = ['item_price_level','item_sales_level','item_collected_level','item_pv_level',
                    'user_age_level','user_star_level','shop_review_num_level','shop_review_positive_rate',
                    'shop_star_level','shop_score_service','shop_score_delivery','shop_score_description', 'user_query_hour']

    data_cont = pd.DataFrame(scaler.fit_transform(data[cont_columns]), columns = cont_columns)
    data_cate = data[cate_columns]
    data = pd.concat([data_cate, data_cont, data.hour], axis = 1)

    return data, label, cate_columns, cont_columns


def feature_generate(data):
    print('feature generating...')
    data, label, cate_columns, cont_columns = process_data(data)

    embeddings_tensors = []
    continuous_tensors = []
    for ec in cate_columns:
        layer_name = ec + '_inp'
        # For categorical features, we em-bed the features in dense vectors of dimension 6×(category cardinality)**(1/4)
        embed_dim = data[ec].nunique() if int(6 * np.power(data[ec].nunique(), 1/4)) > data[ec].nunique() \
            else int(6 * np.power(data[ec].nunique(), 1/4))
        t_inp, t_build = embedding_input(layer_name, data[ec].nunique(), embed_dim)
        embeddings_tensors.append((t_inp, t_build))
        del(t_inp, t_build)
    for cc in cont_columns:
        layer_name = cc + '_in'
        t_inp, t_build = continous_input(layer_name)
        continuous_tensors.append((t_inp, t_build))
        del(t_inp, t_build)
    inp_layer =  [et[0] for et in embeddings_tensors]
    inp_layer += [ct[0] for ct in continuous_tensors]
    inp_embed =  [et[1] for et in embeddings_tensors]
    inp_embed += [ct[1] for ct in continuous_tensors]
    return data, label, inp_layer, inp_embed

def embedding_input(name, n_in, n_out):
    inp = Input(shape = (1, ), dtype = 'int64', name = name)
    return inp, Embedding(n_in, n_out, input_length = 1)(inp)

def continous_input(name):
    inp = Input(shape = (1, ), dtype = 'float32', name = name)
    return inp, Reshape((1, 1))(inp)


# The optimal hyperparameter settings were 8 cross layers of size 54 and 6 deep layers of size 292 for DCN
# Embed "Soil_Type" column (embedding dim == 15), we have 8 cross layers of size 29   
def fit(inp_layer, inp_embed, X, y, *params): #X_val,y_val
    print('fitting...')
    #inp_layer, inp_embed = feature_generate(X, cate_columns, cont_columns)
    input = merge(inp_embed, mode = 'concat')
    print('\tinput shape: ', input.shape)
    
    # deep layer
    for i in range(4):
        if i == 0:
            deep = Dense(272, activation='relu')(Flatten()(input))
        else:
            deep = Dense(272, activation='relu')(deep)

    # cross layer
    cross = CrossLayer(output_dim = input.shape[2].value, num_layer = 8, name = "cross_layer")(input)

    #concat both layers
    output = merge([deep, cross], mode = 'concat')
    output = Dense(1, activation = 'sigmoid')(output)
    model = Model(inp_layer, output) 
    
    print(model.summary())
    # plot_model(model, to_file = '/Users/ewenwang/Documents/practice_data/conversion_rate/model.png', show_shapes = True)
    model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ["accuracy"])
    if len(params) == 2:
        X_val = params[0]
        y_val = params[1]    
        model.fit([X[c] for c in X.columns], y, batch_size = 1024, epochs = 5, validation_data = ([X_val[c] for c in X_val.columns], y_val))
    else:
        model.fit([X[c] for c in X.columns], y, batch_size = 1024, epochs = 1)
    return model


# https://keras.io/layers/writing-your-own-keras-layers/
class CrossLayer(layers.Layer):
    def __init__(self, output_dim, num_layer, **kwargs):
        self.output_dim = output_dim
        self.num_layer = num_layer
        super(CrossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        print('building...')
        self.input_dim = input_shape[2]
        self.W = []
        self.bias = []
        for i in range(self.num_layer):
            self.W.append(self.add_weight(shape = [1, self.input_dim], initializer = 'glorot_uniform', name = 'w_' + str(i), trainable = True))
            self.bias.append(self.add_weight(shape = [1, self.input_dim], initializer = 'zeros', name = 'b_' + str(i), trainable = True))
        self.built = True

    def call(self, input):
        for i in range(self.num_layer):
            if i == 0:
                cross = Lambda(lambda x: Add()([K.sum(self.W[i] * K.batch_dot(K.reshape(x, (-1, self.input_dim, 1)), x), 1, keepdims = True), self.bias[i], x]))(input)
            else:
                cross = Lambda(lambda x: Add()([K.sum(self.W[i] * K.batch_dot(K.reshape(x, (-1, self.input_dim, 1)), input), 1, keepdims = True), self.bias[i], input]))(cross)
        return Flatten()(cross)

    def compute_output_shape(self, input_shape):
        return (None, self.output_dim)

    

if __name__ == "__main__":
    # data download from https://www.kaggle.com/uciml/forest-cover-type-dataset/data
    wd = '/Users/ewenwang/Documents/practice_data/conversion_rate/'
    file = ['round2_keras_train_7.txt', 'round2_keras_test_a.txt']

    print('loading...')
    data = pd.read_csv(wd+file[0],sep= " ")
    test = pd.read_csv(wd+file[1],sep = " ")
    data = pd.concat([data,test], keys=['train', 'test'])
    # data.to_csv(wd+'round2_keras.txt', index=False, sep=' ')
    gc.collect()
    X, y, inp_layer, inp_embed = feature_generate(data)
    del data
    gc.collect()
    
    online = True 
    if online == False:
        X = pd.concat([X, y], axis = 1)
        X_train = X.loc[X.hour < 11] 
        X_test = X.loc[X.hour == 11]  
        del X_train['hour']
        del X_test['hour']
        y_train = X_train.is_trade
        del X_train['is_trade']
        y_test =  X_test.is_trade
        del X_test['is_trade']
    else:
        X = pd.concat([X, y], axis = 1)
        del X['hour']
        X_train = X.loc[X.is_trade.notnull()]
        X_test = X.loc[X.is_trade.isnull()]
        y_train = X_train.is_trade
        del X_train['is_trade']
        y_test =  X_test.is_trade
        del X_test['is_trade']

    print('training...')
    if online == False:
        model = fit(inp_layer, inp_embed, X_train, y_train, X_test, y_test)
        
        val_pre = model.predict([X_train[c] for c in X_train.columns],batch_size=1024)[:,0]
        print("train log_loss",log_loss(y_train.values,val_pre))
        
        val_pre = model.predict([X_test[c] for c in X_test.columns],batch_size=1024)[:,0]
        print("test log_loss",log_loss(y_test.values,val_pre))

    else:
        model = fit(inp_layer, inp_embed, X_train, y_train)
        val_pre = model.predict([X_test[c] for c in X_test.columns],batch_size=1024)[:,0]
        test['predicted_score'] = val_pre
        test[['instance_id', 'predicted_score']].to_csv(wd+'results.txt', index=False,sep=' ')#保存在线提交结果