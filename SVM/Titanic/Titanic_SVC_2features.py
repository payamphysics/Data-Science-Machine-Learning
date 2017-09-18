
# coding: utf-8

# In[75]:

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


# In[76]:

# The training data
train = pd.read_csv('C://Users//Payam//Documents//1_Data_Science//Titanic//train.csv')
train.head()


# In[77]:

# Contains only the test features and not the target 
test = pd.read_csv('C://Users//Payam//Documents//1_Data_Science//Titanic//test.csv')
test.head()


# In[78]:

# Contains only the test features and not the target 
test = pd.read_csv('C://Users//Payam//Documents//1_Data_Science//Titanic//test.csv')
test.head()


# In[79]:

# Contains the survival data for the target set
target = pd.read_csv('C://Users//Payam//Documents//1_Data_Science//Titanic//gender_submission.csv')
target.head()
# Attaching the 'Survived Column to the test set
test['Survived'] = target.Survived


# In[80]:


target.head()


# In[81]:

# Checks to see if there's any null entries
train.Pclass.isnull().any()


# In[82]:

train.isnull().any()


# In[83]:

# Filling in the null values with the average value of the column (only one of the possible approaches)
train.Age[train.Age.isnull()] = train.Age.mean()


# In[84]:

train.Age.isnull().any()


# In[85]:

# to test the balance of the data
train.groupby('Survived').size()
# the data is not balanced in terms of the target


# In[86]:

# choosing a balanced training set.
train_bal = train.groupby('Survived').apply(lambda x: x.sample(342))


# In[87]:

#train_bal.head()
#train_bal.groupby('Survived').size()
train_bal.groupby(train_bal.Survived).size()


# In[88]:

# Checks to see if there's any null entries
test.isnull().any()


# In[89]:

# Filling in the null values with the average value of the columns
test.Age[test.Age.isnull()] = test.Age.mean()


# In[90]:

# Looking at the distict levels of the Sex column
train_bal.Sex.unique()


# In[91]:

# Producing a dictionary to replace levels with numbers
keys = {}
for i in range(len(train_bal.Sex.unique())):
    keys[train_bal.Sex.unique()[i]] = i*100

keys


# In[92]:

# Replacing levels with numbers
train_bal.Sex.replace(keys, inplace = True)
test.Sex.replace(keys, inplace = True)


# In[93]:

# Looking at the distict levels of the Sex column
train_bal.Sex.unique()


# In[94]:

# Arranging train and test data to be used
data_train_bal = train_bal[['Sex','Age']]
target_train_bal = train_bal.Survived
data_test = test[['Sex','Age']]
target_test = test.Survived


# In[95]:

# Normalizing some of the numerical columns
#data_train_bal[['Age','Fare']] = preprocessing.normalize(data_train_bal[['Age','Fare']])
#data_test[['Age','Fare']] = preprocessing.normalize(data_test[['Age', 'Fare']])


# In[96]:

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


# In[97]:

# SVM on the train_bal set and the prediction. Parameters found from the grid search.
clf=SVC(C=10000.0,kernel='rbf', gamma=0.001, class_weight=None, probability=True)
clf.fit(data_train_bal, target_train_bal)
Probabilities=clf.predict_proba(data_test)


# In[98]:

# Assessing the performance
y_true, y_pred = target_test, clf.predict(data_test)
print(classification_report(y_true, y_pred))


# In[99]:

accuracy_score(y_true, y_pred, normalize=False, sample_weight=None)


# In[100]:

accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)


# In[101]:

# Explicit way of assessing the performance
diff = [abs(x-y) for (x,y) in zip([list(x).index(max(x)) for x in Probabilities], list(target_test))]
#diff
# Shows nuber of correct (0) predictions and wrong predictions
results = pd.Series(diff).groupby(pd.Series(diff)).size();results


# In[102]:

correct = results[0]

try:
    incorrect = results[1]
except KeyError:
        incorrect = 0

success_rate = correct/(correct+incorrect);success_rate


# In[103]:

print(correct)


# In[104]:

# Probabilities
plt.rc('figure', figsize=(20, 12))
plt.plot(range(len(Probabilities)), [max(x) for x in Probabilities])
plt.plot(range(len(Probabilities)), [min(x) for x in Probabilities])
plt.tick_params(labelsize=30)
plt.tick_params(labelsize=30, length=15, width = 2, pad = 10)
plt.show()


# In[105]:

xx = np.linspace(0, 90, 100)
np.shape(xx)
yy = np.linspace(-10, 110, 100)
xx, yy = np.meshgrid(xx, yy)
Xfull = np.c_[xx.ravel(), yy.ravel()]
np.shape(Xfull)


# In[106]:

y_pred2 = clf.predict(data_train_bal)
# View probabilities
probas = clf.predict_proba(Xfull)
n_classes = np.unique(y_pred2).size
for k in range(n_classes):
    plt.subplot(1,n_classes, k + 1)
    plt.title("Class %d" % k, fontsize=35)
    
    imshow_handle = plt.imshow(probas[:, k].reshape((100, 100)), extent=(-10, 110, 0, 90), origin='lower')
    plt.xticks(())
    plt.yticks(())
    idx = (y_pred2 == k)
    Xtrain = np.array(data_train_bal)
    if idx.any():
        plt.scatter(Xtrain[idx, 0], Xtrain[idx, 1], marker='o', c = 'k')
    ax = plt.axes([0.15, 0.04, 0.7, 0.05])
    plt.title("Probability", fontsize=35)
    plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')

ax = plt.axes([0.15, 0.04, 0.7, 0.05])
font_size = 34 # Adjust as appropriate.
plt.title("Probability", fontsize=35)
cbar = plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')
cbar.ax.tick_params(labelsize=30)
plt.show()


# In[ ]:



