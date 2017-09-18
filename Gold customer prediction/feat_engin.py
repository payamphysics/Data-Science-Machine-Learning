# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 20:41:01 2017

@author: Payam
"""
customer['age'] = [datetoage(x) if x is not np.nan else np.nan for x in customer.birth]

spend['datediff'] = spend.datemax - spend.spend_day

spend.head()

spend_trim = spend[['id', 'spend_type', 'spend']][spend.datediff < 121]
spend_trim.head()

count_g = spend_trim[spend_trim.spend_type == 1].groupby('id').size()
count_s = spend_trim[spend_trim.spend_type == 2].groupby('id').size()
count_o = spend_trim[spend_trim.spend_type == 3].groupby('id').size()
spend_g = spend_trim[spend_trim.spend_type == 1].groupby('id')['spend'].sum()
spend_s = spend_trim[spend_trim.spend_type == 2].groupby('id')['spend'].sum()
spend_o = spend_trim[spend_trim.spend_type == 3].groupby('id')['spend'].sum()
spend_total = spend_trim.groupby('id')['spend'].sum()


spend_std_total = spend_trim.groupby('id')['spend'].std()
spend_max_total = spend_trim.groupby('id')['spend'].max()
spend_med_total = spend_trim.groupby('id')['spend'].median()


spend_gdf = pd.DataFrame({'count_g':count_g, 'spend_g':spend_g})
spend_gdf.reset_index(inplace=True)
spend_gdf.head()

spend_sdf = pd.DataFrame({'count_s':count_s, 'spend_s':spend_s})
spend_sdf.reset_index(inplace=True)

spend_odf = pd.DataFrame({'count_o':count_o, 'spend_o':spend_o})
spend_odf.reset_index(inplace=True)
spend_odf.head()

spend_tot = pd.DataFrame({'spend_total':spend_total})
spend_tot.reset_index(inplace=True)
spend_std = pd.DataFrame({'spend_std_total': spend_std_total})
spend_std.reset_index(inplace=True)
spend_max = pd.DataFrame({'spend_max_total': spend_max_total})
spend_max.reset_index(inplace=True)
spend_med = pd.DataFrame({'spend_med_total': spend_med_total})
spend_med.reset_index(inplace=True)



spend_unq = spend_gdf.merge(spend_sdf, how = 'left', on='id').\
merge(spend_odf, how = 'left', on='id').merge(spend_tot, how = 'left', on='id').\
merge(spend_std, how = 'left', on='id').merge(spend_max, how = 'left', on='id')\
.merge(spend_med, how = 'left', on='id')

spend_unq.columns

spend_unq['count_total'] = spend_unq[['count_g', 'count_s', 'count_o']].sum(axis=1)
spend_unq['spend_avg_g'] = spend_unq['spend_g']/spend_unq['count_g']
spend_unq['spend_avg_o'] = spend_unq['spend_o']/spend_unq['count_o']
spend_unq['spend_avg_s'] = spend_unq['spend_s']/spend_unq['count_s']
spend_unq['spend_avg_total'] = spend_unq['spend_total']/spend_unq['count_total']


merged_data_total = customer[['id', 'postcode', 'age', 'Lan_spoken', \
'prin_payment', 'income', 'sex', 'credit_flag', 'mail_flag', 'business_flag']]\
.merge(spend_unq, on = 'id', how = 'inner').merge(coupon_unq, on = 'id', how = 'inner')\
.merge(service_unq, on = 'id', how = 'inner').merge(demo, left_on ='postcode', right_on = 'POSTCODE', how = 'inner')\
.merge(target[target.id.notnull()] , how='inner', on= 'id')

merged_data_total.shape

[(x, merged_data_total[x].dtype) for x in merged_data_total.columns if merged_data_total[x].dtype == 'O']

merged_data_total = merged_data_total.drop(['id', 'postcode', 'POSTCODE'], axis = 1)

merged_data_total['sex'] = chartonum(merged_data_total,'sex') #{'F': 1, nan: 2, 'M': 3}
merged_data_total['credit_flag'] = chartonum(merged_data_total,'credit_flag')
merged_data_total['mail_flag'] = chartonum(merged_data_total,'mail_flag')
merged_data_total['business_flag'] = chartonum(merged_data_total,'business_flag')
merged_data_total['Lan_spoken'] = chartonum(merged_data_total,'Lan_spoken')
# {nan: 1, 'Y': 2, 'N': 3}

merged_data_total['income'] = [1 if x < 25000 else 2 if (x>= 25000 and x <35000) \
else 3 if (x>= 35000 and x <45000) else 4 if x >= 45000 \
else 0 for x in merged_data_total['income']]


[x for x in merged_data_total.columns if merged_data_total[x].isnull().any()]
len([x for x in merged_data_total.columns if merged_data_total[x].isnull().any()])


merged_data_total.isnull().any()

merged_data_total.head()

repl_mis(merged_data_total,'age', merged_data_total.age.mean())
repl_mis(merged_data_total,'prin_payment', merged_data_total.prin_payment.mean())
repl_mis(merged_data_total,'count_s', 0)
repl_mis(merged_data_total,'spend_s', 0)
repl_mis(merged_data_total,'count_o', 0)
repl_mis(merged_data_total,'spend_o', 0)
repl_mis(merged_data_total,'spend_std_total', 0)
repl_mis(merged_data_total,'spend_avg_o', 0)
repl_mis(merged_data_total,'spend_avg_s', 0)

merged_data_total.isnull().any()

merged_data_total.isnull().any().sum()

cols = [(i, x, merged_data_total[x].dtype) for i,x in zip(range(len(merged_data_total)),merged_data_total.columns)]

merged_data_total.shape

merged_data_total.head()
