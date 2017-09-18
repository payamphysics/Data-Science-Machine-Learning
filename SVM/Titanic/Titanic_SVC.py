
# coding: utf-8

# In[201]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.svm import SVC, libsvm
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import random


# In[202]:

# The training data
train = pd.read_csv('C://Users//Payam//Documents//1_Data_Science//Titanic//train.csv')
train.head()


# In[203]:

# Contains only the test features and not the target 
test = pd.read_csv('C://Users//Payam//Documents//1_Data_Science//Titanic//test.csv')
test.head()


# In[204]:

# Contains only the test features and not the target 
test = pd.read_csv('C://Users//Payam//Documents//1_Data_Science//Titanic//test.csv')
test.head()


# In[205]:

# Contains the survival data for the target set
target = pd.read_csv('C://Users//Payam//Documents//1_Data_Science//Titanic//gender_submission.csv')
target.head()
# Attaching the 'Survived Column to the test set
test['Survived'] = target.Survived


# In[206]:


target.head()


# In[207]:

# Checks to see if there's any null entries
train.Pclass.isnull().any()


# In[208]:

train.isnull().any()


# In[209]:

# Filling in the null values with the average value of the column (only one of the possible approaches)
train.Age[train.Age.isnull()] = train.Age.mean()


# In[210]:

train.Age.isnull().any()


# In[211]:

# to test the balance of the data
train.groupby('Survived').size()
# the data is not balanced in terms of the target


# In[212]:

# choosing a balanced training set.
train_bal = train.groupby('Survived').apply(lambda x: x.sample(342))


# In[213]:

#train_bal.head()
#train_bal.groupby('Survived').size()
train_bal.groupby(train_bal.Survived).size()


# In[214]:

# Checks to see if there's any null entries
test.isnull().any()


# In[215]:

# Filling in the null values with the average value of the columns
test.Age[test.Age.isnull()] = test.Age.mean()
test.Fare[test.Fare.isnull()] =test.Fare.mean()


# In[216]:

# Looking at the distict levels of the Sex column
train_bal.Sex.unique()


# In[217]:

# Producing a dictionary to replace levels with numbers
keys = {}
for i in range(len(train_bal.Sex.unique())):
    keys[train_bal.Sex.unique()[i]] = i+1

keys


# In[218]:

# Replacing levels with numbers
train_bal.Sex.replace(keys, inplace = True)
test.Sex.replace(keys, inplace = True)


# In[219]:

# Looking at the distict levels of the Sex column
train_bal.Sex.unique()


# In[220]:

train_bal.Cabin.unique()


# In[221]:

test.Cabin.unique()


# In[222]:

# Arranging train and test data to be used
data_train_bal = train_bal[['Pclass','Sex','Age','SibSp','Parch','Fare']]
target_train_bal = train_bal.Survived
data_test = test[['Pclass','Sex','Age','SibSp','Parch','Fare']]
target_test = test.Survived


# In[223]:

# Normalizing some of the numerical columns
#data_train_bal[['Age','Fare']] = preprocessing.normalize(data_train_bal[['Age','Fare']])
#data_test[['Age','Fare']] = preprocessing.normalize(data_test[['Age', 'Fare']])


# In[224]:

np.shape(data_train_bal)
#np.shape(target_train_bal)
#np.shape(data_test)
#np.shape(target_test)


# In[119]:

# finding optimum hyperparameters for the SVC method
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],'C': [1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5]}]
scores = ['precision', 'recall']     

for score in scores:
    print("Tuning hyper-parameters for %s" % score)

    model = GridSearchCV(SVC(C=1), tuned_parameters, cv=None, scoring='%s_weighted' % score)
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


# In[225]:

# SVM on the train_bal set and the prediction. Parameters found from the grid search.
clf=SVC(C=10000.0,kernel='rbf', gamma=0.001, class_weight=None, probability=True)
clf.fit(data_train_bal, target_train_bal)
Probabilities=clf.predict_proba(data_test)


# In[228]:

# Assessing the performance
y_true, y_pred = target_test, clf.predict(data_test)
print(classification_report(y_true, y_pred))


# In[229]:

accuracy_score(y_true, y_pred, normalize=False, sample_weight=None)


# In[230]:

accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)


# In[231]:

# Explicit way of assessing the performance
diff = [abs(x-y) for (x,y) in zip([list(x).index(max(x)) for x in Probabilities], list(target_test))]
#diff
# Shows nuber of correct (0) predictions and wrong predictions
results = pd.Series(diff).groupby(pd.Series(diff)).size();results


# In[232]:

correct = results[0]

try:
    incorrect = results[1]
except KeyError:
        incorrect = 0

success_rate = correct/(correct+incorrect);success_rate


# In[233]:

print(correct)


# In[235]:

# probabilities
plt.rc('figure', figsize=(20, 12))
plt.plot(range(len(Probabilities)), [max(x) for x in Probabilities])
plt.plot(range(len(Probabilities)), [min(x) for x in Probabilities])
plt.tick_params(labelsize=30)
plt.tick_params(labelsize=30, length=15, width = 2, pad = 10)
plt.show()


# In[ ]:



