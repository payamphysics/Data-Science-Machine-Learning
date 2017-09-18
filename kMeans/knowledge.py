# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 12:29:33 2017

@author: Payam
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap

 

know_in = pd.read_csv('C:/Users/Payam/Documents/0_MetroC/My_Teaching/kMeans/knowledge.csv')
know_in.head()


know = np.array(know_in)
know = pd.DataFrame(know)
know.head()

list(range(5))

cols = []
for i in range(5):
    cols.append(i)
    rem_cols = list(set(range(5))-set(cols))
    for j in rem_cols:        
        print(i,j)
        plt.figure(figsize=(20, 20))
        plt.scatter(know[i][know[5] == 'very_low'],know[j][know[5] == 'very_low'], s=200)
        plt.scatter(know[i][know[5] == 'Low'],know[j][know[5] == 'Low'], s=200)
        plt.scatter(know[i][know[5] == 'Middle'],know[j][know[5] == 'Middle'], s=200)
        plt.scatter(know[i][know[5] == 'High'],know[j][know[5] == 'High'], s=200)
        plt.show()


cols = []
for i in range(5):
    cols.append(i)
    rem_cols = list(set(range(5))-set(cols))
    for j in rem_cols:
        features = know[[i,j]]
        kmn = KMeans(n_clusters=4, random_state=10)
        kmn.fit(features)
        cents = pd.DataFrame(kmn.cluster_centers_)
        print(i,j)
        plt.figure(figsize=(20, 20))
        plt.scatter(know[i][know[5] == 'very_low'],know[j][know[5] == 'very_low'], s=200)
        plt.scatter(know[i][know[5] == 'Low'],know[j][know[5] == 'Low'], s=200)
        plt.scatter(know[i][know[5] == 'Middle'],know[j][know[5] == 'Middle'], s=200)
        plt.scatter(know[i][know[5] == 'High'],know[j][know[5] == 'High'], s=200)
        plt.scatter(cents[0],cents[1], s=400, color = 'black')
        plt.show()


features = know[[4,4]]

kmn = KMeans(n_clusters=4, random_state=10)
kmn.fit(features)
cents = pd.DataFrame(kmn.cluster_centers_)
plt.figure(figsize=(20, 20))
plt.scatter(know[4][know[5] == 'very_low'],know[4][know[5] == 'very_low'], s=200)
plt.scatter(know[4][know[5] == 'Low'],know[4][know[5] == 'Low'], s=200)
plt.scatter(know[4][know[5] == 'Middle'],know[4][know[5] == 'Middle'], s=200)
plt.scatter(know[4][know[5] == 'High'],know[4][know[5] == 'High'], s=200)
plt.scatter(cents[0],cents[1], s=400, color = 'black')


y = know[5].copy()

h = .001  # step size in the mesh

keys = {}
for i in range(len(y.unique())):
    keys[y.unique()[i]] = i+1

keys

# Replacing levels with numbers
y.replace(keys, inplace = True)

# Create color maps
#cmap_bold = ListedColormap(['#0000ff','#9400d3', '#2e8b57', '#8b4513'])
cmap_bold = ListedColormap(['#228b22', '#8b4513', '#9400d3','#0000cd'])
cmap_light = ListedColormap(['#87cefa', '#9acd32', '#cd853f', '#dda0dd'])


# we create an instance of Neighbours Classifier and fit the data.
kmn = KMeans(n_clusters=4, random_state=10, max_iter=700)
kmn.fit(features)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = know[4].min() - 0.1, know[4].max() + 0.1
y_min, y_max = know[4].min() - 0.1, know[4].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = kmn.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.rc('figure', figsize=(20, 12))
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(know[4], know[4], c=y, cmap=cmap_bold, s = 150)
plt.scatter(cents[0],cents[1], s=500, color = 'black')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()   


know[6] = y
know[[4,5,6]]


features = know[[3,4]]

kmn = KMeans(n_clusters=4, random_state=10)
kmn.fit(features)
kmn.labels_
kmn.cluster_centers_
cents = pd.DataFrame(kmn.cluster_centers_)

plt.figure(figsize=(20, 20))
plt.scatter(know[3][know[5] == 'very_low'],know[4][know[5] == 'very_low'], s=200)
plt.scatter(know[3][know[5] == 'Low'],know[4][know[5] == 'Low'], s=200)
plt.scatter(know[3][know[5] == 'Middle'],know[4][know[5] == 'Middle'], s=200)
plt.scatter(know[3][know[5] == 'High'],know[4][know[5] == 'High'], s=200)
plt.scatter(cents[0],cents[1], s=400, color = 'black')


# Create color maps
cmap_bold = ListedColormap(['#0000ff', '#2e8b57', '#8b4513', '#9400d3'])
cmap_light = ListedColormap(['#87cefa', '#9acd32', '#cd853f', '#dda0dd'])


# we create an instance of Neighbours Classifier and fit the data.
kmn = KMeans(n_clusters=4, random_state=10, max_iter=700)
kmn.fit(features)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = know[3].min() - 0.1, know[3].max() + 0.1
y_min, y_max = know[4].min() - 0.1, know[4].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = kmn.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.rc('figure', figsize=(20, 12))
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(know[3], know[4], c=y, cmap=cmap_bold)
plt.scatter(cents[0],cents[1], s=400, color = 'black')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()       

