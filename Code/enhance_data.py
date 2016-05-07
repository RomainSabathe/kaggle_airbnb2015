import pandas as pd
import numpy as np

""" SETTINGS """
is_training_data = True
complete_enhancing = True

""" CODE """
""" Recovering the data """
if complete_enhancing:
    type_of_data = 'base'
else:
    type_of_data = 'enhanced'
if is_training_data:
    dataname = 'learning_data'
else:
    dataname = 'testing_data'

store = pd.HDFStore('../Data/%s_%s.h5' % (type_of_data, dataname))
data = store['data_users']
new_data = data
store.close()

store = pd.HDFStore('../Data/base_sessions.h5')
sessions = store['sessions']
store.close()

""" Dummifying """
to_dummify = ['gender',
             'signup_method',
             'signup_flow',
             'language',
             'affiliate_channel',
             'affiliate_provider',
             'first_affiliate_tracked',
             'signup_app',
             'first_device_type',
             'first_browser',
             ]

new_data = pd.DataFrame(index=data.index)
for feature in to_dummify:
    new_data = pd.concat([new_data, pd.get_dummies(data[feature],
                                                  prefix=feature)],
                         axis = 1)

""" Adding extra columns """
new_data = pd.concat([new_data, data[[col for col in data.columns.values if col not in to_dummify]]],
                     axis=1)
with open('../Data/%s_state_of_completing.txt' % dataname, 'w') as f:
    f.write('Base dummy variables: ok.')


""" Correcting some of the variables """
to_reduce = ['date_account_created_year',
             'timestamp_first_active_year',
             'date_first_booking_year']
for col in to_reduce:
    new_data[col]  = new_data[col] - 2012
with open('../Data/%s_state_of_completing.txt' % dataname, 'a') as f:
    f.write('\nCentering years: ok.')


""" Integrating sessions data """
groups = sessions.groupby(sessions.index)
actions_count = groups.agg('count')['action']
new_data['actions_count'] = actions_count
new_data['actions_count'] = new_data['actions_count'].fillna(0)
with open('../Data/%s_state_of_completing.txt' % dataname, 'a') as f:
    f.write('\nCounting the number of actions per user: ok.')

for col in ['action', 'action_type', 'action_detail', 'device_type']:
    print col
    groups = sessions.groupby([sessions.index, sessions[col]])
    counts = groups.agg('count').iloc[:,0].unstack()
    new_col_names = {old_name: '%s_%s_count' % (col, old_name) \
                    for old_name in counts.columns.values}
    counts = counts.rename(columns=new_col_names)
    counts = counts.fillna(0)
    #new_data = pd.concat([new_data, counts], axis=1)
    new_data = new_data.join(counts, how='left')
    with open('../Data/%s_state_of_completing.txt' % dataname, 'a') as f:
        f.write("\nCounting the number of '%s'  per user: ok." % col)

new_data = new_data.fillna(0)


""" Drop data that is not accessible in the testing set. """
new_data = new_data.drop(['date_first_booking_year',
                          'date_first_booking_month',
                          'date_first_booking_day'], axis=1)
with open('../Data/%s_state_of_completing.txt' % dataname, 'a') as f:
    f.write('\nDrop date first booking: ok.')


""" Adding time component """
secs_elapsed = sessions['secs_elapsed'].replace({'undef':0})
secs_elapsed = secs_elapsed.fillna(0)
groups = secs_elapsed.groupby(sessions.index)
secs_elapsed = groups.agg(np.sum)
new_data['sum_secs_elapsed'] = secs_elapsed
with open('../Data/%s_state_of_completing.txt' % dataname, 'a') as f:
    f.write('\nAdded time component: ok.')


""" Even more dummy variables """
features = ['date_account_created_year',
            'date_account_created_month', 'date_account_created_day',
            'timestamp_first_active_year', 'timestamp_first_active_month',
            'timestamp_first_active_day']
for feature in features:
    new_data = pd.concat([new_data, pd.get_dummies(data[feature],
                                                  prefix=feature)],
                        axis = 1)
with open('../Data/%s_state_of_completing.txt' % dataname, 'a') as f:
    f.write('\nEven more dummy variables: ok.')


""" Some key dates """
condition = (new_data['date_account_created_month'] == 9) & \
            (new_data['date_account_created_day'] >= 20)
new_data.loc[condition, 'end_of_september'] = 1
new_data.loc[condition - True, 'end_of_september'] = 0

condition = (new_data['date_account_created_month'] == 12) & \
            (new_data['date_account_created_day'] >= 17)
new_data.loc[condition, 'around_christmas'] = 1
new_data.loc[condition - True, 'around_christmas'] = 0

condition = (new_data['date_account_created_month'] == 6) | \
            (new_data['date_account_created_month'] == 7)
new_data.loc[condition, 'pre_summer'] = 1
new_data.loc[condition - True, 'pre_summer'] = 0

condition = (new_data['date_account_created_day'] != \
             new_data['timestamp_first_active_day'])
new_data.loc[condition, 'signin_on_different_day'] = 1
new_data.loc[condition - True, 'signin_on_different_day'] = 0

with open('../Data/%s_state_of_completing.txt' % dataname, 'a') as f:
    f.write('\nSomme binary variables regarding the calendar: ok.')

# ASSUMPTION HERE: we do NOT take care of missing values in 'secs_elapsed'
#sub_sessions = sessions[sessions['secs_elapsed'] != 'undef']
#groups = sub_sessions.groupby(sub_sessions.index)
#time_data = groups.agg(['mean', 'var'])
#new_data = pd.concat([new_data, time_data], axis=1)
#with open('../Data/%s_state_of_completing.txt' % dataname, 'a') as f:
#    f.write('\nAdding mean and var waiting time per user: ok.')

""" Saving """
if is_training_data:
    store = pd.HDFStore('../Data/enhanced_learning_restricted_data.h5')
else:
    store = pd.HDFStore('../Data/enhanced_testing_data.h5')
store['data_users'] = new_data
store.close()

