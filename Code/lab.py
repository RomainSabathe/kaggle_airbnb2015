import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools as it
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import train_test_split, KFold
from sklearn.metrics import confusion_matrix
from score import calc_score
from toolbox import transform_to_submission_format, get_feature_importances
from random import choice
import xgboost as xgb

plt.style.use('ggplot')

""" Recovering the data """
print 'Recovering the data'
store = pd.HDFStore('../Data/enhanced_learning_restricted_data.h5')
data = store['data_users']
data = data.fillna(0)
store.close()

# Deleting a part of the data
#base_size = 30000
#data_country_7 = data[data['country_destination'] == 7]
#_, rows_to_delete = train_test_split(data_country_7, train_size=base_size, random_state=1)
#data = data.drop(rows_to_delete.index, axis=0)
#
#data_country_10 = data[data['country_destination'] == 10]
#_, rows_to_delete = train_test_split(data_country_10, train_size=int(base_size/1.5), random_state=1)
#data = data.drop(rows_to_delete.index, axis=0)
#
#data_country_11 = data[data['country_destination'] == 11]
#_, rows_to_delete = train_test_split(data_country_11, train_size=base_size/8, random_state=1)
#data = data.drop(rows_to_delete.index, axis=0)

data_learn, data_test = train_test_split(data, test_size=0.3, random_state=2)
data_learn, data_valid = train_test_split(data, test_size=0.7, random_state=2)

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

X_test = data_test
X_test = X_test.drop('country_destination', axis=1)
y_test = pd.DataFrame(index=X_test.index)
y_test['country'] = data_test['country_destination']

kf = KFold(len(data), n_folds=3, random_state=1)

""" Learning """
print 'Learning'


""" #### Test: model parameters #### """
#test_name='model_parameters'
#criterion_choices = ['gini', 'entropy']
#n_estimators_choices = range(1, 750)
#min_samples_split_choices = range(10, 5000)
#max_depth_choices = range(2, 50)
#min_samples_leaf_choices = range(10, 5000)
#
#n_experiments = 1000
#
#criterion_exp = []
#n_estimators_exp = []
#min_samples_split_exp = []
#min_samples_leaf_exp = []
#max_depth_exp = []
#scores = []
#
#for n_experiment in range(n_experiments):
#    criterion_exp.append(choice(criterion_choices))
#    n_estimators_exp.append(choice(n_estimators_choices))
#    min_samples_split_exp.append(choice(max_depth_choices))
#    min_samples_leaf_exp.append(choice(min_samples_leaf_choices))
#    max_depth_exp.append(choice(max_depth_choices))
#
#    classif = RandomForestClassifier(n_estimators=n_estimators_exp[-1],
#                                     criterion=criterion_exp[-1],
#                                     random_state=0,
#                                     min_samples_split=min_samples_split_exp[-1],
#                                     max_depth=max_depth_exp[-1],
#                                     min_samples_leaf=min_samples_leaf_exp[-1],
#                                     n_jobs=-1)
#
#    classif.fit(X_learn, y_learn)
#
#    """ Converting the proba into 5 best guesses """
#    proba_countries = classif.predict_proba(X_valid)
#    find_5_best_countries = lambda x: x.argsort()[-5:][::-1]
#    best_guesses = np.apply_along_axis(find_5_best_countries, 1, proba_countries)
#    predictions = pd.DataFrame(best_guesses, index=y_valid.index)
#
#    print '--------------------'
#    print 'criterion = %s' % criterion_exp[-1]
#    print 'min_samples_split = %s' % min_samples_split_exp[-1]
#    print 'max_depth = %s' % max_depth_exp[-1]
#    print 'min_samples_leaf = %s' % min_samples_leaf_exp[-1]
#    scores.append(calc_score(predictions, y_valid))
#    print 'Score = %s' % scores[-1]
#
#    if n_experiment % 20  == 0 and n_experiment > 0:
#        data_score = pd.DataFrame({'Criterion': criterion_exp,
#                                   'n_estimators': n_estimators_exp,
#                                   'min_samples_split': min_samples_split_exp,
#                                   'max_depth': max_depth_exp,
#                                   'min_samples_leaf': min_samples_leaf_exp,
#                                   'score': scores})
#
#        data_score.to_csv('../Lab/%s.csv' % test_name)


""" #### Test: number of features #### """
#test_name='number_features'
#scores = []
#
#classif_base = RandomForestClassifier(n_estimators=186,
#                                 criterion='entropy',
#                                 random_state=0,
#                                 min_samples_split=30,
#                                 max_depth=16,
#                                 min_samples_leaf=11,
#                                 n_jobs=-1)
#classif_base.fit(X_learn, y_learn)
#
#fi = [(name, value) for (name,value) in zip(X_learn.columns.values.tolist(),
#                                            classif_base.feature_importances_)]
#fi = sorted(fi, key=lambda x: x[1], reverse=True)
#features = [f[0] for f in fi]
#features_to_keep = features[:200]
#
#""" Plotting figure importances """
##fi_ = [x[1] for x in fi]
##plt.bar(range(len(fi_)), fi_)
##print features[:10]
##plt.show()
#
#with open('../Data/features_to_keep.dt', 'w') as f:
#    pickle.dump(features_to_keep, f)
#
#for n_features in range(1, len(features)):
#    classif = RandomForestClassifier(**classif_base.get_params())
#
#    X_learn_ = X_learn[features[:n_features]]
#    X_valid_ = X_valid[features[:n_features]]
#    classif.fit(X_learn_, y_learn)
#
#    """ Converting the proba into 5 best guesses """
#    proba_countries = classif.predict_proba(X_valid_)
#    find_5_best_countries = lambda x: x.argsort()[-5:][::-1]
#    best_guesses = np.apply_along_axis(find_5_best_countries, 1, proba_countries)
#    predictions = pd.DataFrame(best_guesses, index=y_valid.index)
#
#    print '--------------------'
#    print 'n_features = %s' % n_features
#    scores.append(calc_score(predictions, y_valid))
#    print 'Score = %s' % scores[-1]
#
#    if n_features % 5  == 0:
#        data_score = pd.DataFrame({'n_features': range(n_features),
#                                   'score': scores})
#
#        data_score.to_csv('../Lab/%s.csv' % test_name)


""" Test: simple test """

#with open('../Data/features_to_keep.dt', 'r') as f:
#    features_to_keep = pickle.load(f)

scores = []
#for train,test in kf:
for _ in range(1):
    #X_learn, X_valid, y_learn, y_valid = X.iloc[train], X.iloc[test], \
    #                                     y.iloc[train], y.iloc[test]
    #y_valid = pd.DataFrame({'country': y_valid})
    #y_test = pd.DataFrame({'country': y_test})

    """ RANDOM FOREST """
    classif_base = RandomForestClassifier(n_estimators=300,
                                     criterion='entropy',
                                     random_state=0,
                                     min_samples_split=1000,
                                     max_depth=10,
                                     min_samples_leaf=100,
                                     n_jobs=-1)
    classif = RandomForestClassifier(**classif_base.get_params())

    """ GRADIENT BOOSTING """
    #classif_base = GradientBoostingClassifier(loss='deviance',
    #                                          learning_rate=0.25,
    #                                          n_estimators=20,
    #                                          max_depth=5,
    #                                          min_samples_split=50,
    #                                          min_samples_leaf=100,
    #                                          random_state=0,
    #                                          verbose=True)
    #classif = GradientBoostingClassifier(**classif_base.get_params())

    """ XGBOOST """
    xg_train = xgb.DMatrix(X_learn, label=y_learn)
    xg_valid = xgb.DMatrix(X_valid, label=y_valid)
    xg_test = xgb.DMatrix(X_test, label=y_test)

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

    watchlist = [ (xg_train, 'train'), (xg_valid, 'test') ]
    num_boost_round = 20
    early_stopping_rounds = 5
    classif = xgb.train(param, xg_train, early_stopping_rounds=early_stopping_rounds,
                        num_boost_round=num_boost_round,
                        evals=watchlist)
    #proba_countries = classif.predict( xg_test, ntree_limit=classif.best_ntree_limit )

    #classif.fit(X_learn, y_learn)

    """ Converting the proba into 5 best guesses """
    #proba_countries = classif.predict_proba(X_valid_)
    def score(X, y):
        X = xgb.DMatrix(X, label=y)
        proba_countries = classif.predict( X, ntree_limit=classif.best_ntree_limit )
        #proba_countries = classif.predict_proba(X)
        find_5_best_countries = lambda x: x.argsort()[-5:][::-1]
        best_guesses = np.apply_along_axis(find_5_best_countries, 1, proba_countries)
        predictions = pd.DataFrame(best_guesses, index=y.index)

        print calc_score(predictions, y)

    score(X_learn, y_learn)
    score(X_valid, y_valid)
    score(X_test, y_test)
    #print np.array(get_feature_importances(classif, X_learn_)[:20])
    #import pdb; pdb.set_trace()
    #miss_rows = predictions[predictions[0] != y_valid['country']]
    #miss_rows = pd.concat([y_valid.loc[miss_rows.index], miss_rows], axis=1)
    #confmat = confusion_matrix(miss_rows.iloc[:,0], miss_rows.iloc[:,1])

    #miss_rows_710 = miss_rows[(miss_rows['country']==10) & (miss_rows[0]==7)]
    #import pdb; pdb.set_trace()

#print '----------------------'
#print 'Mean score = %s' % np.mean(scores)
#print 'Std score = %s' % np.std(scores)

