# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 17:07:51 2017

@author: Payam
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

r1 = [random.uniform(0,1) for i in range(1000)]
t1 = [random.uniform(0,2*np.pi) for i in range(1000)]

px = [r*np.cos(t) for (r,t) in zip(r1,t1)]
py = [r*np.sin(t) for (r,t) in zip(r1,t1)]

r2 = [random.uniform(0,1) for i in range(1000)]
t2 = [random.uniform(0,2*np.pi) for i in range(1000)]

qx = [1+r*np.cos(t) for (r,t) in zip(r2,t2)]
qy = [1+r*np.sin(t) for (r,t) in zip(r2,t2)]

r3 = [random.uniform(0,1) for i in range(1000)]
t3 = [random.uniform(0,2*np.pi) for i in range(1000)]

mx = [r*np.cos(t) for (r,t) in zip(r2,t2)]
my = [1.6+r*np.sin(t) for (r,t) in zip(r2,t2)]


plt.scatter(px,py)
plt.scatter(qx,qy)
plt.scatter(mx,my)

px.extend(qx)
px.extend(mx)
len(px)
py.extend(qy)
py.extend(my)

dicp = {'xcor':px, 'ycor':py}
dfp = pd.DataFrame.from_dict(dicp)



kmn = KMeans(n_clusters=3, random_state=10)
kmn.fit(dfp)
kmn.labels_
kmn.cluster_centers_
cents = pd.DataFrame(kmn.cluster_centers_)



dfp['label'] = kmn.labels_

dfp_zero = dfp[dfp.label == 0]
dfp_one = dfp[dfp.label == 1]
dfp_two = dfp[dfp.label == 2]

plt.scatter(px,py)
plt.scatter(qx,qy)
plt.scatter(mx,my)

plt.scatter(dfp_zero.xcor,dfp_zero.ycor)
plt.scatter(dfp_one.xcor,dfp_one.ycor)
plt.scatter(dfp_two.xcor,dfp_two.ycor)
plt.scatter(cents[0],cents[1], s=400)
