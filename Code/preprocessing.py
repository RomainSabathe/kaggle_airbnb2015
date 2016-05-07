import pandas as pd
import numpy as np
import pickle
import tables
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

plt.style.use('ggplot')

""" SETTINGS """
is_training_data = True

""" CODE """
if is_training_data:
    filename='train_users_2'
else:
    filename = 'test_users'

data = pd.read_csv('../Data/%s.csv' % filename, index_col='id')

""" Handling missing values """
data['age'] = data['age'].fillna(-1)

""" Time series management """
data['date_account_created'] = pd.to_datetime(data['date_account_created'], format='%Y-%m-%d')
data['timestamp_first_active'] = pd.to_datetime(data['timestamp_first_active'], format='%Y%m%d%H%M%S')
data['date_first_booking'] = pd.to_datetime(data['date_first_booking'], format='%Y-%m-%d')

""" Encoding """
# Encoding the labeled features
#to_encode = ['gender',
#             'signup_method',
#             'language',
#             'affiliate_channel',
#             'affiliate_provider',
#             'first_affiliate_tracked',
#             'signup_app',
#             'first_device_type',
#             'first_browser',
#             ]
#if is_training_data:
#    to_encode.append('country_destination')
#    # As for the record, writing which variables are dummies and which are not.
#    with open('../Encoding/variables_types.txt', 'w') as f:
#        f.write('DUMMY VARIABLES:\n')
#        f.write('\n'.join(to_encode))
#        f.write('\n\nQUANTITATIVE VARIABLES:\n')
#        f.write('\n'.join([col for col in data.columns.values if col not in to_encode]))
#
#for feature in to_encode:
#    data[feature] = data[feature].fillna('undef')
#    le = pickle.load(open('../Encoding/LabelEncoder_%s.md' % feature, 'r'))
#    data[feature] = le.transform(data[feature])

# Encoding the dates
data['date_account_created_year']  = data['date_account_created'].map(lambda x: x.year)
data['date_account_created_month'] = data['date_account_created'].map(lambda x: x.month)
data['date_account_created_day']   = data['date_account_created'].map(lambda x: x.day)

data['timestamp_first_active_year']  = data['timestamp_first_active'].map(lambda x: x.year)
data['timestamp_first_active_month'] = data['timestamp_first_active'].map(lambda x: x.month)
data['timestamp_first_active_day']   = data['timestamp_first_active'].map(lambda x: x.day)

data['date_first_booking_year']  = data['date_first_booking'].map(lambda x: x.year)
data['date_first_booking_month'] = data['date_first_booking'].map(lambda x: x.month)
data['date_first_booking_day']   = data['date_first_booking'].map(lambda x: x.day)

data = data.drop(['date_account_created', 'timestamp_first_active', 'date_first_booking'], axis=1)
data = data.fillna(-1)

""" Saving """
if is_training_data:
    store = pd.HDFStore('../Data/base_learning_data.h5')
else:
    store = pd.HDFStore('../Data/base_testing_data.h5')
store['data_users'] = data
store.close()

#""" Some plots """
## 1 - Number of created accounts over the weeks
#kw = lambda x: x.isocalendar()[:2]
#counts = data.groupby(data['date_account_created'].map(kw)).agg('count')
#counts['age'].plot()
#
#plt.show()
