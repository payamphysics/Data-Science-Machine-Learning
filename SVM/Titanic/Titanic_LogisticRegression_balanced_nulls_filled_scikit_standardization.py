# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 14:53:52 2017

@author: Payam
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
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


# Checks to see if there's any null entries
train.Sex.isnull().any()
train.Pclass.isnull().any()
train.SibSp.isnull().any()
train.Parch.isnull().any()
train.Embarked.isnull().any()
train.Fare.isnull().any()
train.Age.isnull().any() # Contains Null


train.Age[train.Age.isnull()] = train.Age.mean()

# to test the balance of the data
train.groupby('Survived').size()
# the data is not balanced in terms of the target

# choosing a balanced training set.
train_bal = train.groupby('Survived').apply(lambda x: x.sample(342))
train_bal.groupby('Survived').size()


# Checks to see if there's any null entries
test.Sex.isnull().any()
test.Pclass.isnull().any()
test.SibSp.isnull().any()
test.Parch.isnull().any()
test.Embarked.isnull().any()
test.Fare.isnull().any() # Contains Null
test.Age.isnull().any() # Contains Null

test.Age[test.Age.isnull()] = test.Age.mean()
test.Fare[test.Fare.isnull()] =test.Fare.mean()

test.groupby('Survived').size()

keys = {}
for i in range(len(train_bal.Sex.unique())):
    keys[train_bal.Sex.unique()[i]] = i+1

keys

# and replaces them with numbers
train_bal.Sex.replace(keys, inplace = True)
test.Sex.replace(keys, inplace = True)

keys = {}
for i in range(len(train_bal.Embarked.unique())):
    keys[train_bal.Embarked.unique()[i]] = i+1

keys
train_bal.Embarked.replace(keys, inplace = True)

train_bal.Embarked.unique()

keys = {}
for i in range(len(test.Embarked.unique())):
    keys[test.Embarked.unique()[i]] = i+1

keys
test.Embarked.replace(keys, inplace = True)


data_train_bal = train_bal[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
target_train_bal = train_bal.Survived
data_test = test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
target_test = test.Survived

data_train_bal[['Age','SibSp','Parch','Fare']] = preprocessing.normalize(data_train_bal[['Age','SibSp','Parch','Fare']])
data_test[['Age','SibSp','Parch','Fare']] = preprocessing.normalize(data_test[['Age','SibSp','Parch','Fare']])


np.shape(data_train_bal)
np.shape(target_train_bal)
np.shape(data_test)
np.shape(target_test)

tuned_parameters = [{'tol': [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],'C': [1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5]}]
scores = ['precision', 'recall']     

for score in scores:
    print("Tuning hyper-parameters for %s" % score)

    model = GridSearchCV(LogisticRegression(tol=1.0, C=1.0), tuned_parameters, cv=None, scoring='%s_weighted' % score)
    model.fit(data_train_bal, target_train_bal)

    print("Best parameters set found on development set:")
    print(model.best_params_)
    print("Grid scores on development set:")
    for params, mean_score, scores in model.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"% (mean_score, scores.std() * 2, params))
        
    print("Detailed classification report:")
    print("The model is train_baled on the full development set.")
    print("The scores are computed on the full evaluation set.")
    y_true, y_pred = target_test, model.predict(data_test)
    print(classification_report(y_true, y_pred))

lgr = LogisticRegression(penalty='l2', dual=False, tol=0.01, C=10.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
lgr.fit(data_train_bal, target_train_bal)
P=lgr.predict_proba(data_test)

result_set = pd.DataFrame(columns = ['pred','target'])
result_set.pred = pd.Series([list(x).index(max(x)) for x in P])
result_set.target = pd.Series(target_test)
result_set.head()
cor_1 = len(result_set.pred[result_set.pred == 1][result_set.target == 1]); cor_1
all_pred_1 = len(result_set.pred[result_set.pred == 1]); all_pred_1
all_target_1 = len(result_set.target[result_set.target == 1]); all_target_1
cor_0 = len(result_set.pred[result_set.pred == 0][result_set.target == 0]); cor_0
all_pred_0 = len(result_set.pred[result_set.pred == 0]); all_pred_0
all_target_0 = len(result_set.target[result_set.target == 0]); all_target_0

precision_1 = cor_1/all_pred_1; precision_1
precision_0 = cor_0/all_pred_0; precision_0

recall_1 = cor_1/all_target_1; recall_1
recall_1 = cor_0/all_target_0; recall_0

[(x,y) for (x,y) in zip([list(x).index(max(x)) for x in P], list(target_test))]
diff = [x-y for (x,y) in zip([list(x).index(max(x)) for x in P], list(target_test))]
#diff
# Shows nuber of correct (0) predictions and wrong predictions
results = pd.Series(diff).groupby(pd.Series(diff)).size();results

correct = results[0]

try:
    incorrect = results[1]+results[-1]
except KeyError:
    try:
        try:
            incorrect = results[1]
        except KeyError:
            incorrect = results[-1]
    except KeyError:
        incorrect = 0

incorrect    

success_rate = correct/(correct+incorrect);success_rate

plt.rc('figure', figsize=(20, 12))
plt.plot(range(len(P)), [max(x) for x in P])
plt.plot(range(len(P)), [min(x) for x in P])
plt.plot(range(len(P)), [1-max(x)-min(x) for x in P])
plt.tick_params(labelsize=30)
plt.tick_params(labelsize=30, length=15, width = 2, pad = 10)
