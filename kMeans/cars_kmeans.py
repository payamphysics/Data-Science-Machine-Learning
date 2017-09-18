# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:07:25 2017

@author: Payam
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random

cars = pd.read_csv('C://Users//Payam//Documents//0_MetroC//Z_My_Teaching//kMeans//Cars.csv')
cars.head()
cars['buying'].unique()

feat = list(set(cars.columns) - {cars.columns[-1]})

features = cars[feat]
features.head()


classes = cars[cars.columns[-1]]

classes.groupby(classes).size()

for i in features.columns:
    if np.dtype(features[i]) == 'O':
        keys = {}
        print(features[i].unique())
        for j in range(len(features[i].unique())):
            keys[features[i].unique()[j]] = j+1
        features[i].replace(keys, inplace = True)
        
features.head()

feat_rand = features.copy()

for i in features.columns:
    feat_rand[i] = feat_rand[i].apply(lambda x: x+random.random()/2)


np.dtype(features['persons'])
features['maint'].unique()

plt.rc('figure', figsize=(20, 12)); plt.scatter(feat_rand.buying, feat_rand.lug_boot)
plt.rc('figure', figsize=(20, 12)); plt.scatter(feat_rand.buying, feat_rand.maint)
plt.rc('figure', figsize=(20, 12)); plt.scatter(feat_rand.buying, feat_rand.persons)

cols = []
for i in features.columns:
    cols.append(i)
    rem_cols = list(set(features.columns)-set(cols))
    for j in rem_cols:        
        print(i,j)
        plt.rc('figure', figsize=(20, 12)); plt.scatter(feat_rand[i], feat_rand[j])
        plt.show()
    


kmn = KMeans(n_clusters=4, random_state=10)
kmn.fit(feat_rand)
kmn.labels_
pd.Series(kmn.labels_).unique()

pd.Series(kmn.labels_).groupby(pd.Series(kmn.labels_)).size()

kmn.cluster_centers_

kmn.predict([[0, 0], [4, 4]])

