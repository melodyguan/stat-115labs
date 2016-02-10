
# coding: utf-8

# In[76]:

import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# In[23]:

train = pd.read_csv("train_2.csv")
train.head()


# In[24]:

## REMOVE THESE LINES OF CODE WHEN RUNNING ON FULL DATASET
train_sample = train.sample(frac = .1)
train_sample.shape


# In[57]:

## we need to drop any rows that have na
train_clean = train_sample.replace([np.inf,-np.inf],np.nan).dropna()
train_clean.shape


# In[58]:

feats = [i for i in range(2,258)]
[feats.append(i) for i in range(260,270)];


# In[59]:

## start separating out the features and the results
## also, get a testing set
X = train_clean[feats]
y = train_clean["gap"]
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.2)


# In[60]:

## normalize the continous features
## keep track of means and stds in order to use them to normalize the testing set
mean = []
std = []

for i in X_train.iloc[:,256:266].columns:
    mean.append(X_train[i].mean())
    std.append(X_train[i].std())


# In[67]:

for i in range(0,10):
    X_train[X_train.iloc[:,256:266].columns[i]] = (X_train[X_train.iloc[:,256:266].columns[i]] - mean[i])/std[i]


# In[72]:

X_train.shape


# In[99]:

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
    #print "The best score is," best_score


# In[98]:

clfrand_forest([1,2,3,4],X_train,y_train)


# In[100]:

## Testing it out with our best estimator
clf = RandomForestClassifier(n_estimators = 2)
clf.fit(X_train,y_train)


# In[101]:

## test on test data but first we must normalize it as well
for i in range(0,10):
    X_test[X_train.iloc[:,256:266].columns[i]] = (X_test[X_train.iloc[:,256:266].columns[i]] - mean[i])/std[i]


# In[102]:

## Testing the model
prediction = clf.predict(X_test)
(mean_squared_error(y_test,prediction))**.5


# In[ ]:



