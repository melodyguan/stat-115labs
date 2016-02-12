
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm
from statsmodels.formula.api import glm, ols
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomTreesEmbedding


# In[2]:

train = pd.read_csv("train_final.csv")
test = pd.read_csv("test_final.csv")

train.drop(['Unnamed: 0', 'Unnamed: 0.1','Unnamed: 0.1'], inplace = True, axis =1)
test.drop(['Unnamed: 0', 'Unnamed: 0.1','Unnamed: 0.1'], inplace = True, axis =1)


# In[6]:

#train.head()


# In[7]:

#train.ix[:,-12:].head()


# In[3]:

# Break it into test and train
itrain, itest = train_test_split(xrange(train.shape[0]), train_size=0.9)
mask=np.ones(train.shape[0], dtype='int')
mask[itrain]=1
mask[itest]=0
mask=(mask==1)
train_train = train[mask]
train_test = train[~mask]


# In[4]:

train_train = train_train.drop(['smiles'], 1)
train_test = train_test.drop(['smiles'], 1)


# In[5]:

features = list(train_train.columns)
features.remove('gap')

X_train = np.asmatrix(train_train[features].astype(np.float64))
X_test = np.asmatrix(train_test[features].astype(np.float64))
y_train = np.asmatrix(train_train['gap'].apply(lambda val: val)).T
y_test = np.asmatrix(train_test['gap'].apply(lambda val: val)).T


# In[19]:

vdict = {}
rdict = {}
for a in [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]:
    for cv_train, cv_test in KFold(y_train.size, 5):
        lin_model_r = Ridge(alpha=a)
        lin_model_r.fit(X_train[cv_train], y_train[cv_train])
        y_pred = np.asmatrix(lin_model_r.predict(X_train[cv_test])).T
        mse = (mean_squared_error(y_train[cv_test], y_pred.T)**(0.5))
        vdict[a] = mse
        rdict[a] = lin_model_r
        
best_a = min(vdict, key=vdict.get)
best_ridge_model =rdict[best_a]

print 'Cross validation MSE values: ' + str(vdict)
print 'Best a from cross validation: ' + str(best_a)


# In[20]:

y_pred = np.asmatrix(best_ridge_model.predict(X_test)).T
mse = (mean_squared_error(y_test, y_pred.T)**(0.5))

print 'RMSE for Ridge Regression: ' + str(mse)


# In[21]:

vdict = {}
rdict = {}
for a in [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]:
    for cv_train, cv_test in KFold(y_train.size, 5):
        lin_model_r = Lasso(alpha=a)
        lin_model_r.fit(X_train[cv_train], y_train[cv_train])
        y_pred = np.asmatrix(lin_model_r.predict(X_train[cv_test])).T
        mse = (mean_squared_error(y_train[cv_test], y_pred)**(0.5))
        vdict[a] = mse
        rdict[a] = lin_model_r
        
best_a = min(vdict, key=vdict.get)
best_lasso_model =rdict[best_a]

print 'Cross validation MSE values: ' + str(vdict)
print 'Best a from cross validation: ' + str(best_a)


# In[22]:

y_pred = np.asmatrix(best_lasso_model.predict(X_test)).T
mse = (mean_squared_error(y_test, y_pred)**(0.5))

print 'RMSE for Lasso Regression: ' + str(mse)


# In[25]:

vdict = {}
rdict = {}
for a in [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]:
    for cv_train, cv_test in KFold(y_train.size, 5):
        lin_model_r = ElasticNet(alpha=a,l1_ratio=0.5, )
        lin_model_r.fit(X_train[cv_train], y_train[cv_train])
        y_pred = np.asmatrix(lin_model_r.predict(X_train[cv_test])).T
        mse = (mean_squared_error(y_train[cv_test], y_pred)**(0.5))
        vdict[a] = mse
        rdict[a] = lin_model_r
        
best_a = min(vdict, key=vdict.get)
best_elastic_model =rdict[best_a]

print 'Cross validation MSE values: ' + str(vdict)
print 'Best a from cross validation: ' + str(best_a)


# In[26]:

y_pred = np.asmatrix(best_elastic_model.predict(X_test)).T
mse = (mean_squared_error(y_test, y_pred)**(0.5))

print 'RMSE for Elastic Net: ' + str(mse)


# In[ ]:

def convertDataNeuralNetwork(x, y):
    colx = 1 if len(np.shape(x))==1 else np.size(x, axis=1)
    coly = 1 if len(np.shape(y))==1 else np.size(y, axis=1)
    
    fulldata = pybrain.datasets.ClassificationDataSet(colx,coly, nb_classes=2)
    for d, v in zip(x, y):
        fulldata.addSample(d, v)
    
    return fulldata

regressionTrain = convertDataNeuralNetwork(train_data, train_values)

regressionTest = convertDataNeuralNetwork(test_data, test_values)

fnn = FeedForwardNetwork()

inLayer = LinearLayer(regressionTrain.indim)
hiddenLayer = LinearLayer(5)
outLayer = GaussianLayer(regressionTrain.outdim)

fnn.addInputModule(inLayer)
fnn.addModule(hiddenLayer)
fnn.addOutputModule(outLayer)

in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

fnn.addConnection(in_to_hidden)
fnn.addConnection(hidden_to_out)

fnn.sortModules()

trainer = BackpropTrainer(fnn, dataset=regressionTrain, momentum=0.1, verbose=True, weightdecay=0.01)

for i in range(10):

    trainer.trainEpochs(5)

    res = trainer.testOnClassData(dataset=regressionTest )

    print res


# In[6]:

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
    print "The optimal n_estimator, best_score is ",best_param , best_score

rand_forest([1,2,3,4],X_train,y_train)


# In[ ]:



