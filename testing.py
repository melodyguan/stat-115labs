import numpy as np
import pandas as pd

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
print "reading"

train = pd.read_csv("train_final.csv")

test = pd.read_csv("test_final.csv")

print "read"

train.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1'], inplace = True, axis =1)

test.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1'], inplace = True, axis =1)

features = train.iloc[:,1:32].columns

features = features.append(train.iloc[:,33:46].columns)

X = train[features]

y = train['gap']

features_norm = ['ExactMolWt', 'arom_ring_count', 'het_atoms',

       'num_rotate_bonds', 'h_donors', 'h_acceptors', 'fr_NHO', 'fr_NH1',

       'fr_benzene', 'SiH2', 'se', 'n', 's']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
mean = []
std = []

for i in features_norm:
	mean.append(X_train[i].mean())
	std.append(X_train[i].std())

for i in range(0,len(features_norm)):
    X_train[features_norm[i]] = (X_train[features_norm[i]] - mean[1])/std[i]
    X_test[features_norm[i]] = (X_test[features_norm[i]] - mean[1])/std[i]

clf = RandomForestClassifier(n_estimators = 40)
clf.fit(X_train,y_train)
prediction = clf.predict(X_test)
print (mean_squared_error(y_test,prediction))**.5


