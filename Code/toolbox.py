import pandas as pd
import numpy as np
import pickle

def transform_to_submission_format(predictions, index):
    """
    Converts a numpy array of n rows and 5 columns into
    a a single column DataFrame with n*5 rows.
    This format is used for the Kaggle competition.
    """

    new_index = []
    countries = []
    for id in index:
        new_index.extend([id] * 5)
        countries.extend(predictions.loc[id])

    return pd.DataFrame({'country':countries}, index=new_index)


def transform_to_score_format(predictions):
    score_format = pd.DataFrame(index = pd.unique(predictions.index))
    for rank in range(5):
        num_index = np.arange(rank, len(predictions), 5)
        score_format['Pred_%s' % rank] = predictions.iloc[num_index]

    return score_format


def tell_story_session(data_session_user):
    # Loading the encoders
    to_decode = ['action', 'action_type', 'action_detail']
    encoders = {}
    for feature in to_decode:
        encoders[feature] = pickle.load(open('../Encoding/LabelEncoder_%s.md' % feature, 'r'))

    # Decoding the session
    for k in range(len(data_session_user)):
        line = data_session_user.iloc[k]
        decoded = {feature: encoders[feature].inverse_transform(line[feature]) \
                   for feature in to_decode}
        space1 = ' ' * (10 - len(decoded['action_type']))
        space2 = ' ' * (30 - len(decoded['action']))
        print '%s:%s %s%s (%s)' % (decoded['action_type'], space1,
                               decoded['action'], space2,
                               decoded['action_detail'])


def get_feature_importances(classif, data_learn):
    perf = zip(data_learn.columns.values.tolist(), classif.feature_importances_)
    perf = sorted(perf, key=lambda x: x[1], reverse=True)

    return perf
