# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 11:59:09 2017

@author: Payam
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.svm import SVC, libsvm
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
import random

train = pd.read_csv('C://Users//Payam//Documents//1_Data_Science//Titanic//train.csv')
train.head()
# Contains only the test features and not the target 
test = pd.read_csv('C://Users//Payam//Documents//1_Data_Science//Titanic//test.csv')
test.head()
# Contains the survival data for the target set
target = pd.read_csv('C://Users//Payam//Documents//1_Data_Science//Titanic//gender_submission.csv')
target.head()
test['Survived'] = target.Survived

# to test the balance of the data
train.groupby('Survived').size()
test.groupby('Survived').size()
target.groupby('Survived').size()
# the data is not balanced in terms of the target

# Checks to see if there's any null entries
train.Sex.isnull().any()
train.Pclass.isnull().any()
train.SibSp.isnull().any()
train.Parch.isnull().any()
train.Embarked.isnull().any()
train.Fare.isnull().any()
train.Age.isnull().any() # Contains Null

train = train[train.Age.notnull()]

# Checks to see if there's any null entries
test.Sex.isnull().any()
test.Pclass.isnull().any()
test.SibSp.isnull().any()
test.Parch.isnull().any()
test.Embarked.isnull().any()
test.Fare.isnull().any() # Contains Null
test.Age.isnull().any() # Contains Null

test = test[test.Age.notnull()]
test = test[test.Fare.notnull()]

keys = {}
for i in range(len(train.Sex.unique())):
    keys[train.Sex.unique()[i]] = i+1

keys

# and replaces them with numbers
train.Sex.replace(keys, inplace = True)
test.Sex.replace(keys, inplace = True)

keys = {}
for i in range(len(train.Embarked.unique())):
    keys[train.Embarked.unique()[i]] = i+1

keys
train.Embarked.replace(keys, inplace = True)

keys = {}
for i in range(len(test.Embarked.unique())):
    keys[test.Embarked.unique()[i]] = i+1

keys
test.Embarked.replace(keys, inplace = True)

train.Age = [(x-min(train.Age))/(max(train.Age)-min(train.Age)) for x in train.Age]
train.Fare = [(x-min(train.Fare))/(max(train.Fare)-min(train.Fare)) for x in train.Fare]
train.SibSp = [(x-min(train.SibSp))/(max(train.SibSp)-min(train.SibSp)) for x in train.SibSp]
train.Parch = [(x-min(train.Parch))/(max(train.Parch)-min(train.Parch)) for x in train.Parch]

test.Age = [(x-min(test.Age))/(max(test.Age)-min(test.Age)) for x in test.Age]
test.Fare = [(x-min(test.Fare))/(max(test.Fare)-min(test.Fare)) for x in test.Fare]
test.SibSp = [(x-min(test.SibSp))/(max(test.SibSp)-min(test.SibSp)) for x in test.SibSp]
test.Parch = [(x-min(test.Parch))/(max(test.Parch)-min(test.Parch)) for x in test.Parch]


data_train = train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
target_train = train.Survived
data_test = test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
target_test = test.Survived

np.shape(data_train)
np.shape(target_train)
np.shape(data_test)
np.shape(target_test)


#plt.scatter(train.Age, train.Fare)
#plt.rc('figure', figsize=(20, 12))
#plt.tick_params(labelsize=30)
#plt.tick_params(labelsize=30, length=15, width = 2, pad = 10)

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],'C': [1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5]}]
scores = ['precision', 'recall']     

for score in scores:
    print("Tuning hyper-parameters for %s" % score)

    model = GridSearchCV(SVC(C=1), tuned_parameters, cv=None, scoring='%s_weighted' % score)
    model.fit(data_train, target_train)

    print("Best parameters set found on development set:")
    print(model.best_params_)
    print("Grid scores on development set:")
    for params, mean_score, scores in model.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"% (mean_score, scores.std() * 2, params))
        
    print("Detailed classification report:")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    y_true, y_pred = target_test, model.predict(data_test)
    print(classification_report(y_true, y_pred))

# SVM on the train set and the prediction. Parameters found from the grid search.
clf=SVC(C=100000.0,kernel='rbf', gamma=0.01, class_weight=None, probability=True)
clf.fit(data_train, target_train)
P=clf.predict_proba(data_test)
np.shape(P)

#[(x,y) for (x,y) in zip([list(x).index(max(x)) for x in P], list(target_test))]
diff = [x-y for (x,y) in zip([list(x).index(max(x)) for x in P], list(target_test))]
#diff
# Shows number of correct (0) predictions and wrong predictions
results = pd.Series(diff).groupby(pd.Series(diff)).size();results

correct = results[0]
incorrect = results[1]+results[-1] 

success_rate = correct/(correct+incorrect);success_rate

plt.rc('figure', figsize=(20, 12))
plt.plot(range(len(P)), [max(x) for x in P])
plt.plot(range(len(P)), [min(x) for x in P])
plt.plot(range(len(P)), [1-max(x)-min(x) for x in P])
plt.tick_params(labelsize=30)
plt.tick_params(labelsize=30, length=15, width = 2, pad = 10)
