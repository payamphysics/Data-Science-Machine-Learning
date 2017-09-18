# -*- coding: utf-8 -*-
"""
Created on Sat Apr 01 12:40:35 2017

@author: Payam

This procedure reads in two csv files, converts string input into numbers and
saves the outcome as a csv file
"""




import pandas as pn
import numpy as np
from sklearn import svm
import random
import matplotlib.pyplot as plt
from sklearn.svm import SVC, libsvm 
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import random

#file1 = pn.read_csv('C:\\Users\Payam\\Documents\\1_Data_Science\\i_360\\i_360_assignment\\File1.csv')

#print file1

#print len(file1)

""" This was exteremely slow
file1.insert(3, 'NumResponse', 0)
for i in range(len(file1)-1):
    print i
    if file1['SPENDINGRESPONSE'][i] == 'Spend to Improve Economy':
        file1['NumResponse'][i] = 1
    elif file1['SPENDINGRESPONSE'][i] == 'Reduce National Debt and Deficit':
        file1['NumResponse'][i] = 0
    
print file1
"""

file1 = pn.read_csv('C:\\Users\Payam\\Documents\\1_Data_Science\\i_360\\i_360_assignment\\File1.csv')

print file1

file1_mat = np.array(file1)
#file1_mat=file1_1.values


file1_num = file1_mat


for i in range(len(file1)):
    if file1_num[i][2] == 'Spend to Improve Economy':        
        file1_num[i][2] = 1
    elif file1_num[i][2] == 'Reduce National Debt and Deficit':
        file1_num[i][2] = 0
            
#print file1_num

file2 = pn.read_csv('C:\\Users\Payam\\Documents\\1_Data_Science\\i_360\\i_360_assignment\\File2.csv')

#print file2

file2_mat = np.array(file2)


n_col= file2_mat.shape[1]


file2_mat_mod_2 = file2_mat

""" This shows that the headers are not read from the csv file"""
print file2_mat[0][0] 

for i in range(n_col):
    unique = []
    k = 0    
    for j in range(20000):
        if type(file2_mat[j][i]) == str:
            if file2_mat[j][i] in unique:
                file2_mat_mod_2[j][i] = k
            elif file2_mat[j][i] not in unique:
                unique.append(file2_mat[j][i])
                k += 1
                file2_mat_mod_2[j][i] = k
                               

print file2_mat_mod_2[0][0] 


df = pn.DataFrame(file2_mat_mod_2)
df.to_csv("C:\\Users\Payam\\Documents\\1_Data_Science\\i_360\\i_360_assignment\\out.csv")

for i in range(n_col):    
        for j in range(20000):  
            file2_mat_mod_2[j][i] = float(file2_mat_mod_2[j][i])

averages = []
for j in range(n_col):
    s = 0
    c = 0
    for i in range(20000):
        if file2_mat_mod_2[i][j] >= 0 :
            s += file2_mat_mod_2[i][j]
            c += 1
    averages.append(float(s)/float(c)) 
    
#print averages

#print np.isnan(file2_train_x[14999][147])

file2_mat_mod_2_nonan = file2_mat_mod_2

for j in range(1,n_col):
    for i in range(20000):
        if np.isnan(file2_mat_mod_2_nonan[i][j]) == True:
            file2_mat_mod_2_nonan[i][j] = averages[j]


df = pn.DataFrame(file2_mat_mod_2_nonan)
df.to_csv("C:\\Users\Payam\\Documents\\1_Data_Science\\i_360\\i_360_assignment\\out_nonan.csv")



#print max(file2_mat_mod_2_nonan[:,3]), min(file2_mat_mod_2_nonan[:,3])

""" Check this part of the code"""
file2_mat_mod_2_nonan_scal = file2_mat_mod_2_nonan

for j in range(1,n_col):
    mini = min(file2_mat_mod_2_nonan[:,j])
    maxi = max(file2_mat_mod_2_nonan[:,j])
    inter = float(maxi - mini)
    for i in range(20000):
        if inter != 0:
            file2_mat_mod_2_nonan_scal[i][j] = (file2_mat_mod_2_nonan[i][j] - mini)/inter
        else:
            file2_mat_mod_2_nonan_scal[i][j] = file2_mat_mod_2_nonan[i][j]/maxi
        
df = pn.DataFrame(file2_mat_mod_2_nonan_scal)
df.to_csv("C:\\Users\Payam\\Documents\\1_Data_Science\\i_360\\i_360_assignment\\out_nonan_scaled.csv")



# the method astype(int) turns float values to int
#file2_train_x = file2_mat_mod_2_nonan_scal[0:12000,1:n_col]
file2_x = file2_mat_mod_2_nonan_scal[0:20001,1:n_col]

lable_y = []
for i in range(20000):
    lable_y.append(file1_num[i][-1])
    
#file1_train_y = lable_y[0:12000]

#print file1_train_y

#file2_test_x = file2_mat_mod_2_nonan_scal[12000:16000,1:n_col]

"""
SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True
, probability=False, tol=0.001, cache_size=200, class_weight=None, 
verbose=False, max_iter=-1, decision_function_shape=None, 
random_state=None) 
"""

data_train, data_test, target_train, target_test = train_test_split(file2_x, lable_y, test_size=0.1, random_state=0)

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],'C': [1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5]}]
scores = ['precision', 'recall']      

for score in scores:
    print("Tuning hyper-parameters for %s" % score)

    model = GridSearchCV(SVC(C=1), tuned_parameters, cv=None, scoring='%s_weighted' % score)
    model.fit(data_train, target_train)

    print("Best parameters set found on development set:")
    print(model.best_params_)
    print("Grid scores on development set:")
    for params, mean_score, scores in model.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"% (mean_score, scores.std() * 2, params))
        
    print("Detailed classification report:")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    y_true, y_pred = target_test, model.predict(data_test)
    print(classification_report(y_true, y_pred))

clf=svm.SVC(C=10,kernel='rbf', gamma=0.01, class_weight=None, probability=True)

clf.fit(data_train, target_train)

# Saving the trained algorithm (i.e. saving the support vectors)
print 'Saving the trained SVM model...' 
joblib.dump(clf,'C:\\Users\Payam\\Documents\\1_Data_Science\\i_360\\My_Work\\trained_SVM_model.pkl')
print 'Done!'

#perdic = clf.predict(data_test)

# Recover the probabilites 'P' of classification of the data_test points and convert to percentages
print 'Predicting the classification probabilites on the testing set...' 
P=clf.predict_proba(data_test)
P=np.array(P[:,1]*100)
print 'Done.. and saving probabilities...' 
np.savetxt('C:\\Users\Payam\\Documents\\1_Data_Science\\i_360\\My_Work\\P_Balanced.txt', P, newline='\n', fmt='%f')
print 'Done!'

# Define the target column 'T' and save it together with P in a 2-column array for comparison
print 'Defining the target column of the testing set to be used for the assessment...'
T=np.array(target_test)
PT=np.vstack([P, T])
PT=np.transpose(PT)
print 'Done and saving probabilities versus the target in a nx2 array...'
np.savetxt('C:\\Users\Payam\\Documents\\1_Data_Science\\i_360\\My_Work\\compare_P_T_Balanced.txt', PT, newline='\n', fmt='%f')
print 'Done!'
np.savetxt('C:\\Users\Payam\\Documents\\1_Data_Science\\i_360\\My_Work\\Test_Target.txt', T, newline='\n', fmt='%f')


# Plot a histogram of the probabilities
print 'Preparing to plot...'
bins=np.linspace(0,100,100)
plt.title("Classification probabilites of the trained data")
plt.xlabel("Probability")
plt.ylabel("Population")

print 'Done and plotting...'
# Plot a histogram of the probabilities
plt.hist(P,bins,facecolor='blue',alpha=0.5)
plt.show()
print 'Done!'

# Assessment: Calculating the accuracy of the model based on the test set
print 'Calculating the accuracy of the model based on the test set...'
l=len(PT)
a=np.zeros(l)
a=list(a)
for i in range(l):
    if (PT[i][0] >= 50 and PT[i][1] ==1) or (PT[i][0] < 50 and PT[i][1] ==0):
       a[i] = 1 
    else:
       a[i] = 0
print 'The trained model is correct', sum(a)/float(l)*100,'% of the time'

#########################################################################################################
#########################################################################################################


"""The result of the above code is 69.2% because the target label is 0 in
69.2% of the cases and the model predicts all zeros, therfore it has the 
same precision, in other words the modeling perfromed here is absolutely 
useless"""


