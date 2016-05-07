import pandas as pd
import numpy as np
import pickle
import tables
from sklearn.preprocessing import LabelEncoder

train_file = 'train_users_2'
test_file = 'test_users'

data_learn = pd.read_csv('../Data/%s.csv' % train_file, index_col='id')
data_test = pd.read_csv('../Data/%s.csv' % test_file, index_col='id')

# Encoding the labeled features
to_encode = ['gender',
             'signup_method',
             'signup_flow',
             'language',
             'affiliate_channel',
             'affiliate_provider',
             'first_affiliate_tracked',
             'signup_app',
             'first_device_type',
             'first_browser',
             'country_destination',
             ]

for feature in to_encode:
    if feature == 'country_destination': # does not exist for the test set
        le = LabelEncoder()
        le.fit(data_learn[feature])
    else:
        data_learn[feature] = data_learn[feature].fillna('undef')
        data_test[feature] = data_test[feature].fillna('undef')

        # Reviewing the possible classes for each features and taking the union of train and learn
        train_classes = pd.unique(data_learn[feature])
        test_classes = pd.unique(data_test[feature])

        classes = set(train_classes) | set(test_classes)
        classes = list(set(train_classes) | set(test_classes))
        classes = pd.Series(classes)

        le = LabelEncoder()
        le.fit(classes)

    pickle.dump(le, open('../Encoding/LabelEncoder_%s.md' % feature, 'w'))

    # Writing down the meaning of all encodings
    with open('../Encoding/%s.txt' % feature, 'w') as f:
        str = '\n'.join(['%s - %s' % (k,klass) for (k,klass) in enumerate(le.classes_)])
        f.write(str)

""" Encoders for the sessions variables. """
store = pd.HDFStore('../Data/sessions.h5')
sessions = store['sessions']
store.close()

to_encode = ['action',
             'action_type',
             'action_detail',
             'device_type',
             ]

for feature in to_encode:
    # Special case of device_type which has 'unknown' instead of NaN
    if feature == 'device_type':
        sessions[feature][sessions[feature] == '-unknown-'] = 'undef'
    sessions[feature] = sessions[feature].fillna('undef')

    # Reviewing the possible classes for each features
    feature_classes = pd.unique(sessions[feature])

    le = LabelEncoder()
    le.fit(feature_classes)

    pickle.dump(le, open('../Encoding/LabelEncoder_%s.md' % feature, 'w'))

    # Writing down the meaning of all encodings
    with open('../Encoding/%s.txt' % feature, 'w') as f:
        str = '\n'.join(['%s - %s' % (k,klass) for (k,klass) in enumerate(le.classes_)])
        f.write(str)

