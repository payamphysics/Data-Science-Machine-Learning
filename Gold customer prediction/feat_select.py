# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 13:49:04 2017

@author: Payam
"""

# Turning merged data to array and back to dataframe
merged_data_total_arr = np.array(merged_data_total)
merged_data_total_arr = pd.DataFrame(merged_data_total_arr)
[(x, merged_data_total_arr[x].dtype) for x in merged_data_total_arr.columns]

merged_data_total_arr.shape

merged_data_total_arr.groupby(168).size()

allcols = list(zip(merged_data_total_arr.columns, merged_data_total.columns))

# defining target
targ = merged_data_total_arr[168]
targ.unique()

# calculating correlation of features with target and keeping higher ones
correlations = []
for i in range(168):
    correlations.append((abs(pearsonr(merged_data_total_arr[i],\
    targ)[0]),pearsonr(merged_data_total_arr[i],targ)[1],i))

correlations
correlations[130] = (0,1.0,130)
len(correlations)
rem_col = [x[2] for x in correlations if x[0] > 0.05]; rem_col
len(rem_col)


# redifining features ******************
feat = merged_data_total_arr[rem_col]
feat = preprocessing.scale(feat)
feat.shape
# Lasso feature selection **************
gridsearch_lasso(feat, targ)
lass = lassoselection(feat,targ, C=0.05, class_weight = {0:0.5, 1:2})

coef = [(abs(x),y) for x,y in zip(lass.coef_[0],rem_col)]; coef
coef.sort()
coef
len(coef)
lis_f = [x[1] for x in coef[len(coef)-6:len(coef)]]; lis_f

# Final features *****************************
feat = merged_data_total_arr[lis_f]
feat_copy = merged_data_total_arr[lis_f]
feat.shape
feat.head()
feat = preprocessing.scale(feat)

targ.groupby(targ).size()
