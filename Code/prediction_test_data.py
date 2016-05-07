import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from score import calc_score
from toolbox import transform_to_submission_format
import xgboost as xgb

classif_name = 'XGBoost_2016-2-3_10h57m'

""" Recovering the data and the classifier """
print 'Recovering the data'
store = pd.HDFStore('../Data/enhanced_testing_data.h5')
data = store['data_users']
data = data.fillna(0)
store.close()


""" Making sure that the number of features in the testing data
    is the same as in the training data.
"""
store = pd.HDFStore('../Data/enhanced_learning_restricted_data.h5')
training_data = store['data_users']
training_data = training_data.drop('country_destination', axis=1)
store.close()

#with open('../Data/features_to_keep.dt', 'r') as f:
#    features_to_keep = pickle.load(f)
#training_data = training_data.loc[:,features_to_keep]

train_columns = training_data.columns
test_columns = data.columns.values
missing_columns = [col for col in train_columns if col not in test_columns]
overflow_columns = [col for col in test_columns if col not in train_columns]

emptyDataFrame = pd.DataFrame(0, columns=missing_columns, index=data.index)
data = pd.concat([data, emptyDataFrame], axis=1)
data = data.drop(overflow_columns, axis=1)

""" Loading the classifier """
#classif = pickle.load(open('../Models/%s.md' % classif_name))
xg_test = xgb.DMatrix(data)
param = {}
param['nthread'] = 4
param['num_class'] = 12

classif = xgb.Booster(param)
classif.load_model('../Models/%s.md' % classif_name)
proba_countries = classif.predict( xg_test, ntree_limit=9 )

print 'Making a prediction'
#proba_countries = classif.predict_proba(data)

print 'Outputting'
find_5_best_countries = lambda x: x.argsort()[-5:][::-1]
best_guesses = np.apply_along_axis(find_5_best_countries, 1, proba_countries)
predictions = pd.DataFrame(best_guesses, index=data.index)

# Generating a proper DataFrame format
le_country = pickle.load(open('../Encoding/LabelEncoder_country_destination.md', 'r'))
one_col_pred_digits = transform_to_submission_format(predictions, data.index)
one_col_pred_names = le_country.inverse_transform(one_col_pred_digits)
final_pred = pd.DataFrame(one_col_pred_names, index=one_col_pred_digits.index)
final_pred.columns = ['country']

final_pred.to_csv('../Predictions/%s.csv' % classif_name, index_label='id')
