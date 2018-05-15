from glob import glob
import re 
import numpy as np
np.random.seed(0) # ensure reproducibility
np.set_printoptions(suppress = True)
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
# Models
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# NN
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
# Stacking
from vecstack import stacking

data = data.merge(pd.read_csv(wd+pagerank_union, sep=' '), on='instance_id', how='left')

n_classes = 2


def build_keras_model_1():
    model = Sequential()
    model.add(Dense(64, 
                    input_dim=X_train.shape[1], 
                    kernel_initializer='normal', 
                    activation='relu'))
    model.add(Dense(n_classes, 
                    kernel_initializer='normal', 
                    activation='softmax'))
    model.compile(optimizer='rmsprop', 
                  loss='categorical_crossentropy', 
                  metrics=['categorical_accuracy'])
    return model

# Caution! All models and parameter values are just 
# demonstrational and shouldn't be considered as recommended.
models_1 = [ 
    GaussianNB(),
    
    LogisticRegression(random_state=0),
    
    ExtraTreesClassifier(random_state=0, n_jobs=-1, n_estimators=100, max_depth=3),
                         
    RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=100, max_depth=3),
        
    XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, n_estimators=100, max_depth=3),
                  
    LGBMClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, n_estimators=100, max_depth=3),
                  
    KerasClassifier(build_fn=build_keras_model_1, epochs=2, batch_size=32, verbose=0)
]

S_train_1, S_test_1 = stacking(models_1,                   # list of models
                               X_train, y_train, X_test,   # data
                               regression=False,           # classification task (if you need 
                                                           #     regression - set to True)
                               mode='oof_pred',            # mode: oof for train set, fit on full 
                                                           #     train and predict test set once
                               needs_proba=True,           # predict probabilities (if you need 
                                                           #     class labels - set to False) 
                               save_dir='.',               # save result and log in current dir 
                                                           #     (to disable saving - set to None)
                               metric=log_loss,            # metric: callable
                               n_folds=5,                  # number of folds
                               stratified=True,            # stratified split for folds
                               shuffle=True,               # shuffle the data
                               random_state=0,             # ensure reproducibility
                               verbose=2)                  # print all info

print('We have %d classes and %d models so in resulting arrays \
we expect to see %d columns.' % (n_classes, len(models_1), n_classes * len(models_1)))
print('S_train_1 shape:', S_train_1.shape)
print('S_test_1 shape: ', S_test_1.shape)

# Our arrays and log were saved in current dir
names = sorted(glob('*.npy'))
npy_1_name = names[0] # for later use

print('Arrays:')
for name in names:
    print(name)

names = sorted(glob('*.log.txt'))
log_1_name = names[0] # for later use

print('\nLogs:')
for name in names:
    print(name)

# Initialize some other 1st level model(s)
def build_keras_model_2():
    model = Sequential()
    model.add(Dense(256, 
                    input_dim=X_train.shape[1], 
                    kernel_initializer='normal', 
                    activation='relu'))
    model.add(Dense(64, 
                    kernel_initializer='normal', 
                    activation='relu'))
    model.add(Dense(n_classes, 
                    kernel_initializer='normal', 
                    activation='softmax'))
    model.compile(optimizer='rmsprop', 
                  loss='categorical_crossentropy', 
                  metrics=['categorical_accuracy'])
    return model

# Caution! All models and parameter values are just 
# demonstrational and shouldn't be considered as recommended.
models_2 = [        
    KerasClassifier(build_fn=build_keras_model_2, epochs=5, 
                    batch_size=32, verbose=0)
]

S_train_2, S_test_2 = stacking(models_2,                   # list of models
                               X_train, y_train, X_test,   # data
                               regression=False,           # classification task (if you need 
                                                           #     regression - set to True)
                               mode='oof_pred',            # mode: oof for train set, fit on full 
                                                           #     train and predict test set once
                               needs_proba=True,           # predict probabilities (if you need 
                                                           #     class labels - set to False) 
                               save_dir='.',               # save result and log in current dir 
                                                           #     (to disable saving - set to None)
                               metric=log_loss,            # metric: callable
                               n_folds=5,                  # number of folds
                               stratified=True,            # stratified split for folds
                               shuffle=True,               # shuffle the data
                               random_state=0,             # ensure reproducibility
                               verbose=2)                  # print all info

names = sorted(glob('*.npy'))

print('Arrays:')
for name in names:
    print(name)
    
names = sorted(glob('*.log.txt'))

print('\nLogs:')
for name in names:
    print(name)

print("Let's open this log: %s" % log_1_name)
with open(log_1_name) as f:
    lines = f.readlines()

print("Let's look what models did we build in those session.\n")
for line in lines:
    if re.search(r'^model [0-9]+', line):
        print(line)


print('We have %d classes and %d models TOTAL so in resulting arrays \
we expect to see %d columns.' % (n_classes, len(models_1) + len(models_2), 
                                 n_classes * (len(models_1) + len(models_2))))

# Create empty arrays
S_train_all = np.zeros((X_train.shape[0], 0))
S_test_all = np.zeros((X_test.shape[0], 0))

# Load results
for name in sorted(glob('*.npy')):
    print('Loading: %s' % name)
    S = np.load(name)
    S_train_all = np.c_[S_train_all, S[0]]
    S_test_all = np.c_[S_test_all, S[1]]
    
print('\nS_train_all shape:', S_train_all.shape)
print('S_test_all shape: ', S_test_all.shape)

# Initialize 2nd level model
model = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, 
                      n_estimators=100, max_depth=3)
    
# Fit 2nd level model
model = model.fit(S_train_all, y_train)

# Predict
y_pred = model.predict_proba(S_test_all)

# Final prediction score
print('Final prediction score: %.8f' % log_loss(y_test, y_pred))
