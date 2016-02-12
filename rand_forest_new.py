import pandas as pd

import numpy as np

from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error



print "reading"

train = pd.read_csv("train_final.csv")

test = pd.read_csv("test_final.csv")

print "read"

#train = train.sample(frac = .1)

train.drop(['Unnamed: 0', 'Unnamed: 0.1',

 'Unnamed: 0.1'], inplace = True, axis =1)

test.drop(['Unnamed: 0', 'Unnamed: 0.1',

'Unnamed: 0.1'], inplace = True, axis =1)



features = train.iloc[:,1:32].columns

features = features.append(train.iloc[:,33:46].columns)



X = train[features]

y = train['gap']

print type(y)

print "type insise y: ", type(y[1])

print "made feature and response matrix"



features_norm = ['ExactMolWt', 'arom_ring_count', 'het_atoms',

       'num_rotate_bonds', 'h_donors', 'h_acceptors', 'fr_NHO', 'fr_NH1',

       'fr_benzene', 'SiH2', 'se', 'n', 's']



X_train, X_test, y_train, y_test = train_test_split(

     X, y, test_size=0.2)

mean = []

std = []

for i in features_norm:

    mean.append(X_train[i].mean())

    std.append(X_train[i].std())



for i in range(0,len(features_norm)):

    X_train[features_norm[i]] = (X_train[features_norm[i]] - mean[1])/std[i]

    X_test[features_norm[i]] = (X_test[features_norm[i]] - mean[1])/std[i]



def rand_forest (params, X, y):

    best_score = 100000000

    best_param = 0

    for i in params:

        clf = RandomForestClassifier(n_estimators = i)

        scores = cross_validation.cross_val_score(clf, X, y, cv= 5, scoring = "mean_squared_error")

        scoring = [abs(k) for k in scores]

        score = np.mean(scoring)

        if score<best_score:

            best_score = score

            best_param = i

            best_clf = clf

    print "The optimat n_estimator, best_score is ",best_param , best_score



print "defined fun"



rand_forest([1,2,3,4],X_train.as_matrix(),y_train.as_matrix())