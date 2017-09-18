# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 11:29:09 2017

@author: Payam
"""

# importance level *********************************
coef_with_names = [(abs(y), allcols[x][1]) for x,y in zip(lis_f, list(lgr.coef_[0]))]
coef_with_names.sort()
coef_with_names

coefs_sort = [x[0] for x in coef_with_names]; coefs_sort
feat_name = [x[1] for x in coef_with_names]; feat_name

plt.figure(figsize=(20, 20))
plt.rcParams['font.size'] = 35
plt.pie(coefs_sort, labels=feat_name, autopct='%1.1f%%')
plt.axis('equal')

# Feature decile charts *********************
train, test_copy, train_target, test_target = train_test_split(feat_copy, targ, test_size=0.3, random_state=10)

featdf = feat_decile(test,test_copy[19],lgr,bins=10)

plt.figure(figsize=(20, 12.34))
plt.xticks(featdf['DECILE_BIN'])
plt.tick_params(labelsize=40)
plt.tick_params(labelsize=40, length=15, width = 2, pad = 10)
plt.xlabel('Deciles', fontsize=35, labelpad=20)
plt.ylabel('Decile Mean', fontsize=35, labelpad=20)
plt.bar(list(range(1,11)), featdf['feat_mean'])
plt.title('spend_avg_g', fontsize=40,  y=1.03)



featdf = feat_decile(test,test_copy[11],lgr,bins=10)

plt.figure(figsize=(20, 12.34))
plt.xticks(featdf['DECILE_BIN'])
plt.tick_params(labelsize=40)
plt.tick_params(labelsize=40, length=15, width = 2, pad = 10)
plt.xlabel('Deciles', fontsize=35, labelpad=20)
plt.ylabel('Decile Mean', fontsize=35, labelpad=20)
plt.bar(list(range(1,11)), featdf['feat_mean'])
plt.title('spend_s', fontsize=40,  y=1.03)



featdf = feat_decile(test,test_copy[14],lgr,bins=10)

plt.figure(figsize=(20, 12.34))
plt.xticks(featdf['DECILE_BIN'])
plt.tick_params(labelsize=40)
plt.tick_params(labelsize=40, length=15, width = 2, pad = 10)
plt.xlabel('Deciles', fontsize=35, labelpad=20)
plt.ylabel('Decile Mean', fontsize=35, labelpad=20)
plt.bar(list(range(1,11)), featdf['feat_mean'])
plt.title('spend_total', fontsize=40,  y=1.03)



featdf = feat_decile(test,test_copy[5],lgr,bins=10)

plt.figure(figsize=(20, 12.34))
plt.xticks(featdf['DECILE_BIN'])
plt.tick_params(labelsize=40)
plt.tick_params(labelsize=40, length=15, width = 2, pad = 10)
plt.xlabel('Deciles', fontsize=35, labelpad=20)
plt.ylabel('Decile Mean', fontsize=35, labelpad=20)
plt.bar(list(range(1,11)), featdf['feat_mean'])
plt.title('credit_flag', fontsize=40,  y=1.03)



featdf = feat_decile(test,test_copy[8],lgr,bins=10)

plt.figure(figsize=(20, 12.34))
plt.xticks(featdf['DECILE_BIN'])
plt.tick_params(labelsize=40)
plt.tick_params(labelsize=40, length=15, width = 2, pad = 10)
plt.xlabel('Deciles', fontsize=35, labelpad=20)
plt.ylabel('Decile Mean', fontsize=35, labelpad=20)
plt.bar(list(range(1,11)), featdf['feat_mean'])
plt.title('count_g', fontsize=40,  y=1.03)



featdf = feat_decile(test,test_copy[10],lgr,bins=10)

plt.figure(figsize=(20, 12.34))
plt.xticks(featdf['DECILE_BIN'])
plt.tick_params(labelsize=40)
plt.tick_params(labelsize=40, length=15, width = 2, pad = 10)
plt.xlabel('Deciles', fontsize=35, labelpad=20)
plt.ylabel('Decile Mean', fontsize=35, labelpad=20)
plt.bar(list(range(1,11)), featdf['feat_mean'])
plt.title('count_s', fontsize=40,  y=1.03)





# lift chary data *****************************************
train, test, train_target, test_target = train_test_split(feat, targ, test_size=0.3, random_state=10)
test.shape

liftdf = calc_lift(test,test_target,lgr,bins=10)
liftdf

liftdf.to_csv('C:/Users/Payam/Desktop/liftdf.csv')

# Cumulative gain chart ****************************************
plt.figure(figsize=(20, 12.34))
plt.xticks(liftdf['PERCENTS'])
plt.tick_params(labelsize=30)
plt.tick_params(labelsize=40, length=15, width = 2, pad = 10)
plt.xlabel('Cumulative % of all customers', fontsize=40, labelpad=20)
plt.ylabel('Cumulative % of true gold customers', fontsize=40, labelpad=20)
plt.grid(b=True, which='both', color='0.65',linestyle='-')
plt.plot(liftdf['PERCENTS'], liftdf['PERCENTS'], marker='o', linestyle='--', markersize = 20, linewidth=5, label='Average')
plt.plot(liftdf['PERCENTS'], liftdf['CUM_GAIN'], marker='o', linestyle='-', markersize = 20, linewidth=5, label='Model')
plt.title('Cumulative Gain Chart', fontsize=40,  y=1.03)
plt.legend(loc=2, fontsize = 40)

# Cumulative Lift chart ****************************************
plt.figure(figsize=(20, 12.34))
plt.xticks(liftdf['PERCENTS'])
plt.tick_params(labelsize=30)
plt.tick_params(labelsize=40, length=15, width = 2, pad = 10)
plt.xlabel('Cumulative % of all customers', fontsize=40, labelpad=20)
plt.ylabel('Cumulative lift', fontsize=40, labelpad=20)
plt.grid(b=True, which='both', color='0.65',linestyle='-')
plt.ylim([0,3])
plt.plot(liftdf['PERCENTS'], liftdf['DECILE_BASE'], marker='o', linestyle='--', markersize = 20, linewidth=5, label='Average')
plt.plot(liftdf['PERCENTS'], liftdf['CUM_LIFT'], marker='o', linestyle='-', markersize = 20, linewidth=5, label='Model')
plt.title('Cumulative Lift Chart', fontsize=40,  y=1.03)
plt.legend(loc=1, fontsize = 40)


# Decile chart ****************************************
plt.figure(figsize=(20, 12.34))
plt.xticks(liftdf['DECILE_BIN'])
plt.tick_params(labelsize=40)
plt.tick_params(labelsize=40, length=15, width = 2, pad = 10)
plt.xlabel('Deciles', fontsize=35, labelpad=20)
plt.ylabel('Decile lift', fontsize=35, labelpad=20)
plt.bar(list(range(0,11)), liftdf['DECILE_LIFT'])
plt.title('Decile Chart', fontsize=40,  y=1.03)

# KS curve ****************************************
plt.figure(figsize=(20, 12.34))
plt.grid(b=True, which='both', color='0.65',linestyle='-')
plt.tick_params(labelsize=30)
plt.tick_params(labelsize=40, length=15, width = 2, pad = 10)
plt.xlabel('Cumulative % of all', fontsize=40, labelpad=20)
plt.ylabel('Cumulative % of __', fontsize=40, labelpad=20)
plt.plot(liftdf['PERCENTS'], liftdf['CUM_NEG'], linewidth=5, label='Non-Gold', marker='o',markersize = 20, linestyle = '--')
plt.plot(liftdf['PERCENTS'], liftdf['CUM_GAIN'], linewidth=5, marker='o',markersize = 20, label='Gold', linestyle = '-')
plt.plot(liftdf['PERCENTS'], liftdf['PERCENTS'], linewidth=5, marker='o',markersize = 20, label='Gold and Non-Gold Average', linestyle = '-')
plt.title('Kolmogorov–Smirnov (KS) Curve', fontsize=40,  y=1.03)
plt.text(10, 80, 'KS_max = 41%', fontsize=40)
plt.legend(loc=4, fontsize = 35)

KSfactor = max(liftdf['CUM_GAIN']-liftdf['CUM_NEG']); KSfactor

# ROC curve ****************************************
train, test, train_target, test_target = train_test_split(feat, targ, test_size=0.3)

lgr = LogisticRegression(class_weight={0 : 0.5, 1: 2})
lgr.fit(train,train_target)
pred = lgr.predict(test)
print(classification_report(test_target, pred))
print('accuracy_score is', accuracy_score(test_target, pred, normalize=True, sample_weight=None))
scores = lgr.predict_proba(test)
print('roc_auc_score is', roc_auc_score(test_target, scores[:,1]))
lgrcoef = lgr.coef_

fpr, tpr, thresholds = roc_curve(test_target, scores[:,1])
roc_names = ['fpr', 'tpr', 'thresholds']
roc_data = [fpr, tpr, thresholds]
rocdf = pd.DataFrame(dict(zip(roc_names, roc_data)))
rocdf


plt.figure()
plt.figure(figsize=(20, 12.34))
lw = 2
plt.plot(rocdf.fpr, rocdf.tpr, color='darkorange',
         lw=lw, label='Model (AUC = %0.2f)' 
         % roc_auc_score(test_target, scores[:,1]), linewidth=5)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', 
         label = 'Average',linewidth=5)
#plt.xticks(rocdf.fpr)
plt.tick_params(labelsize=30)
plt.tick_params(labelsize=40, length=15, width = 2, pad = 10)
plt.xlabel('1-Specificity (False Positive Rate)', fontsize=40, labelpad=20)
plt.ylabel('Sensitivity (True Positive Rate)', fontsize=40, labelpad=20)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=40,  y=1.03)
plt.legend(loc="lower right", fontsize = 40)



