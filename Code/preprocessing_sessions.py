import pandas as pd
import numpy as np
import pickle
import tables
import random
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from toolbox import tell_story_session

plt.style.use('ggplot')

""" CODE """
""" Loading the data """
store = pd.HDFStore('../Data/sessions.h5')
sessions = store['sessions']
store.close()

store = pd.HDFStore('../Data/enhanced_learning_restricted_data.h5')
data_learn = store['data_users']
store.close()
store = pd.HDFStore('../Data/enhanced_testing_data.h5')
data_test = store['data_users']
store.close()

total_data_index = pd.Series(list(set(set(data_learn.index) | set(data_test.index))))
sessions = sessions.loc[total_data_index]

""" Handling missing values """
sessions = sessions.fillna('undef')

""" Encoding """
# Encoding the labeled features
to_encode = ['action',
             'action_type',
             'action_detail',
             'device_type',
             ]
for feature in to_encode:
    # Special case of device_type which has 'unknown' instead of NaN
    if feature == 'device_type':
        sessions[feature][sessions[feature] == '-unknown-'] = 'undef'
    le = pickle.load(open('../Encoding/LabelEncoder_%s.md' % feature, 'r'))
    sessions[feature] = le.transform(sessions[feature])

""" Saving """
store = pd.HDFStore('../Data/base_sessions.h5')
store['sessions'] = sessions
store.close()
