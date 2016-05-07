import numpy as np
import pandas as pd
import random as rd
import timeit

from score import calc_score

data = pd.read_csv('../Data/train_users_2.csv', usecols=['id', 'country_destination'])
data.index = data['id']
id_users = pd.unique(data['id'])[:10000]
targets = pd.DataFrame({'country':data.loc[id_users]['country_destination']}, index=id_users)

""" Generating random predictions """
len_preds = 5
country_pool = ['US', 'FR', 'CA', 'GB', 'ES', 'IT', 'PT', 'NL', 'DE', 'AU', 'NDF', 'other']
dict_pred = {}
for id_user in id_users:
    dict_pred[id_user] = rd.sample(country_pool, len_preds)

predictions = pd.DataFrame(dict_pred)
predictions = predictions.T

start = timeit.default_timer()
print calc_score(predictions, targets)
stop = timeit.default_timer()
print 'Calculated in %s seconds.' % (stop - start)
