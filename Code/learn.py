import pandas as pd
import numpy as np
import pickle
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from score import calc_score
from toolbox import transform_to_submission_format
from sklearn.cross_validation import train_test_split, KFold
import xgboost as xgb

""" Recovering the data """
print 'Recovering the data'
store = pd.HDFStore('../Data/enhanced_learning_restricted_data.h5')
data = store['data_users']
store.close()

data_learn, data_valid = train_test_split(data, test_size=0.2, random_state=1)

X = data
X = X.drop('country_destination', axis=1)
y = data['country_destination']

X_learn = data_learn
X_learn = X_learn.drop('country_destination', axis=1)
y_learn = pd.DataFrame(index=X_learn.index)
y_learn['country'] = data_learn['country_destination']

X_valid = data_valid
X_valid = X_valid.drop('country_destination', axis=1)
y_valid = pd.DataFrame(index=X_valid.index)
y_valid['country'] = data_valid['country_destination']

""" Feature selection """
#print 'Feature selection'
#with open('../Data/features_to_keep.dt', 'r') as f:
#    features_to_keep = pickle.load(f)
#X = X.loc[:,features_to_keep]

""" Learning """
print 'Learning'

""" Random Forest """
#model_name = 'RandomForestClassifier'
#classif = RandomForestClassifier(n_estimators=413,
#                                 criterion='gini',
#                                 random_state=0,
#                                 min_samples_split=36,
#                                 max_depth=42,
#                                 min_samples_leaf=58,
#                                 n_jobs=-1)
#classif.fit(X, y)

"""" XGBoost """
model_name = 'XGBoost'
param = {}
param['objective'] = 'multi:softprob'
param['eta'] = 0.3
param['gamma'] = 5.
param['max_depth'] = 6
param['learning_rate'] = 0.1
param['subsample'] = 1.
param['colsample_bytree'] = 1.
param['min_child_weight'] = 100
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 12

xg_train = xgb.DMatrix(X, label=y)
xg_valid = xgb.DMatrix(X_valid, label=y_valid)
watchlist = [ (xg_train, 'train'), (xg_valid, 'test') ]

num_boost_round = 30
early_stopping_rounds = 3
classif = xgb.train(param, xg_train, early_stopping_rounds=early_stopping_rounds,
                    num_boost_round=num_boost_round,
                    evals=watchlist)
print classif.best_ntree_limit


""" Saving the model """
print 'Saving the model'
current_time = datetime.datetime.now()
date_str = '%s-%s-%s_%sh%sm' % (current_time.year,
                               current_time.month,
                               current_time.day,
                               current_time.hour,
                               current_time.minute)
filename = '%s_%s' % (model_name, date_str)

if model_name == 'RandomForestClassifier':
    pickle.dump(classif, open('../Models/%s.md' % filename, 'w'))
else:
    classif.save_model('../Models/%s.md' % filename)
