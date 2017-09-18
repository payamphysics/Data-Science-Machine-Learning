# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 09:29:20 2017

@author: Payam
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import random
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression

#*********************************************************************
# Running the files which define functions and load data
runfile('C:/Users/Payam/Documents/0_MetroC/0_Final_Projects/Projectj_1/codes/general_functions.py', wdir='C:/Users/Payam/Documents/0_MetroC/0_Final_Projects/Projectj_1/codes')
runfile('C:/Users/Payam/Documents/0_MetroC/0_Final_Projects/Projectj_1/codes/loading_data.py', wdir='C:/Users/Payam/Documents/0_MetroC/0_Final_Projects/Projectj_1/codes')
runfile('C:/Users/Payam/Documents/0_MetroC/0_Final_Projects/Projectj_1/codes/feat_engin.py', wdir='C:/Users/Payam/Documents/0_MetroC/0_Final_Projects/Projectj_1/codes')
runfile('C:/Users/Payam/Documents/0_MetroC/0_Final_Projects/Projectj_1/codes/feat_select.py', wdir='C:/Users/Payam/Documents/0_MetroC/0_Final_Projects/Projectj_1/codes')
#*********************************************************************

# hyper-parameter optimization ***********
gridsearch_model(feat, targ)

# Cross validation ***********************
lgcros = LogisticRegression(class_weight={0 : 0.5, 1: 2})
corsval_auc = cross_val_score(lgcros, feat, targ, cv=5, scoring = 'roc_auc')
corsval_auc
print("AUC: %0.2f (+/- %0.2f)" % (corsval_auc.mean(), corsval_auc.std() * 2))

# model application **********************
lgr = modelapplication(feat,targ, class_weight={0 : 1, 1: 2})

lgr.coef_

# The names and coefficients of the final features
coef_with_names = [(x, allcols[x][1],y) for x,y in zip(lis_f, list(lgr.coef_[0]))]; coef_with_names


#Average AUC of the model ******************************
auc_sc = []
for i in range(100):
    auc = modelapplication_auc(feat,targ, class_weight={0 : 0.5, 1: 2})
    auc_sc.append(auc)

mean_auc = np.array(auc_sc).mean()
std_auc = np.array(auc_sc).std()
print("AUC: %0.2f (+/- %0.2f)" % (mean_auc, std_auc))

# Multicolinearity check **********************
lis_ff = multicolcheck(feat, lis_f); lis_ff # There is no Multicolinearity
