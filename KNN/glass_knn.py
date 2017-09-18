# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:19:54 2017

@author: Payam
"""

import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

gdata = pd.read_csv('C://Users//Payam//Documents//0_MetroC//Z_My_Teaching//KNN//Glass_data.csv') 
gdata.head()
gdata.Glass_type.unique()

# looking at the balance of the target
gdata.groupby('Glass_type').size()

# choosing features and target
ncols = np.shape(gdata)[1]; ncols
features = gdata[gdata.columns[0:ncols-1]]
target = gdata[gdata.columns[ncols-1]]
#target = gdata.Glass_type


# train-test split
data_train, data_test, target_train, target_test = train_test_split(features, target, test_size=0.3, random_state=10)

# the KNN model
n_neigh = 2
knnc = neighbors.KNeighborsClassifier(n_neigh, weights='uniform')
knnc.fit(data_train, target_train)
pred = knnc.predict(data_test)

# looking at the predictions compared to true values
pd.Series(pred).groupby(pd.Series(pred)).size()
target_test.groupby(target_test).size()

# evaluating the predictions
print(classification_report(target_test, pred))
accuracy_score(target_test, pred, normalize=True, sample_weight=None)
