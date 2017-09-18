# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:25:10 2017

@author: Payam
"""

import numpy as np
import pandas as pd
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

wdata= pd.read_csv('C://Users//Payam//Documents//0_MetroC//Z_My_Teaching//KNN//winequality-red.csv')
wdata.head()
np.shape(wdata)

wdata.groupby('quality').size()

# column names
col_names = wdata.columns; col_names

# creating a dictionary for changing column names to numbers
coldic = {}
for i in range(len(col_names)):
    coldic[col_names[i]] = i
coldic

# changing column names to numbers
wdata = wdata.rename(columns=coldic)
wdata.columns

# checking for null values
wdata.isnull().any()

# looking at the balance of the data
wdata.groupby(11).size()
# choosing the more frequent labels (for the sake of simplicity)
wdata = wdata[wdata[11].isin([5,6,7])]
wdata.groupby(11).size()

# choosing features and target
ncols = np.shape(wdata)[1]; ncols
features = wdata[wdata.columns[0:ncols-1]]
target = wdata[wdata.columns[ncols-1]]

# train-test split
data_train, data_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=10)

# the KNN model
n_neigh = 1
knnc = neighbors.KNeighborsClassifier(n_neigh, weights='uniform')
knnc.fit(data_train, target_train)
pred = knnc.predict(data_test)

# looking at the predictions compared to true values
pd.Series(pred).groupby(pd.Series(pred)).size()
target_test.groupby(target_test).size()

# evaluating the predictions
print(classification_report(target_test, pred))
accuracy_score(target_test, pred, normalize=True, sample_weight=None)

# Sequential Feature Selector
sfs1 = SFS(knnc, k_features=11, forward=True, floating=False, verbose=2, scoring='accuracy', cv=5)
sfs1 = sfs1.fit(np.array(data_train), np.array(target_train))

sfs1.subsets_
sfs1.k_feature_idx_
sfs1.k_score_

# choosing the best features
data_train_best = data_train[[0, 1, 10, 7]]
data_test_best = data_test[[0, 1, 10, 7]]

# performing KNN with the best features
knnc.fit(data_train_best, target_train)
pred = knnc.predict(data_test_best)

# evaluating KNN with the best features
print(classification_report(target_test, pred))
accuracy_score(target_test, pred, normalize=True, sample_weight=None)
