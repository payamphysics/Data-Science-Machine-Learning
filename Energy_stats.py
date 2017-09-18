# -*- coding: utf-8 -*-
"""
Created on Sun May 28 10:40:55 2017

@author: Payam
"""

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

###############################################################
# Q1)

################
# a)
# loading the "Energy Indicators.xls" file into a dataframe, by removing the unwanted rows and
# columns and renaming columns
skrows = sum([range(0,11), range(12,18), range(245,284)], [])
energy = pd.read_excel('C:\\Users\\Payam\\Documents\\0_MetroC\\\
Python\Project\\Energy Indicators.xls', parse_cols = [2,3,4,5], \
skiprows= skrows, \
names=['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable'])

for ct in range(227):
    ct_p = re.findall('([A-Za-z ]+) \(\S*', energy.loc[ct,'Country'])
    if len(ct_p) > 0:
        energy.loc[ct, 'Country'] = ct_p[0]
    ct_n = re.findall('([A-Za-z ,]+)[0-9]+', energy.loc[ct, 'Country'])
    if len(ct_n) > 0:
        energy.loc[ct, 'Country'] = ct_n[0]

energy.head(10)

################
# b)
# Converts variable Energy Supply to gigajoules; NA for missing data.
energy['Energy Supply'] = 1000000 * energy['Energy Supply']

energy['Energy Supply'] = [np.nan if type(x) == unicode else x for x in energy['Energy Supply']] 
energy['Energy Supply']          
################
# c) Renaming some countries:
energy['Country'] = energy['Country'].replace({'Republic of Korea':\
'South Korea', 'United States of America':'United States', 'United Kingdom of \
Great Britain and Northern Ireland':'United Kingdom', \
'China, Hong Kong Special Administrative Region':'Hong Kong'})

for i in range(227):
    s = energy['Country'][i]
    s.find(' (')
    if s.find(' (') != -1:
        energy['Country'][i] = s.replace(s[s.find(' ('):], '')
        

energy['Country']

################
# d) loading GDP data; renaming some countries
GDP = pd.read_csv('C:\\Users\\Payam\\Documents\\0_MetroC\\\
Python\Project\\world_bank.csv', skiprows=range(1,4), header = 1)

GDP.head(10)

GDP['Country Name'] = GDP['Country Name'].replace({'Korea, Rep.':\
'South Korea', 'Iran, Islamic Rep.':'Iran','Hong Kong SAR, China':'Hong Kong'})

GDP.columns

################
# e) loading Sciamgo Journal and Country Rank data
ScimEn = pd.read_excel('C:\\Users\\Payam\\Documents\\0_MetroC\\\
Python\Project\\scimagojr-3.xlsx')

ScimEn.columns

################
# f)
# Changing a column name and merging the three dataframe to obtain the
# final dataframe 
GDP = GDP.rename(columns = {'Country Name':'Country'})

cols = sum([[0], range(50,60)], [])

fdata = ScimEn[ScimEn.Rank < 16].merge(energy, on = 'Country')\
                    .merge(GDP[cols], on='Country')

# Sets the names of the countries as the index
fdata = fdata.set_index('Country')
np.shape(fdata)

np.shape(fdata)

################
# Makes a copy of the final data.
final_data = fdata.copy()

final_data.columns

###############################################################
# Q2)
# Takes is a dataframe df; key is the name of the index of the df. newcol is the name of
# the new column to be created. ymin and ymax is the year-range. Returns
# a series containing the average of the quantitied in the range of columns in 
# the year-range with key as its key. 
def avGDP(df, key, newcol, ymin, ymax):
    yr_list = list(map(str, range(ymin,ymax+1)))
    df[newcol] = df[yr_list].mean(axis=1)
    df[key] = df.index  
    aGDP = pd.Series(df[[key, newcol]].sort_values([newcol],\
             ascending = False)[newcol])   
    return aGDP

avgGDP = avGDP(final_data, 'Country', 'Avg_10_year', 2006, 2015)
avgGDP

###############################################################
# Q3)
# takes in dataframe df, and a key to refer to index of avGDP function (which was
# defined above); newcol which is the name of a new column to be created; ymin
# and ymax which indicate the year range; and rank which is the desired rank.
# Returns the GDP of the country with the GDP of specified rank. 
def gdpnth(df, key, newcol, ymin, ymax, rank):
    gd = avGDP(df, key, newcol, ymin, ymax).index[rank-1]
    gd_change = df[str(ymax)][df[key] == gd] - \
       df[str(ymin)][df[key] == gd]
    return float(gd_change) 
    
gdp6th = gdpnth(final_data, 'Country', 'Avg_10_year', 2006, 2015, 6)
gdp6th


###############################################################      
# Q4)
# Takes dataframe df and a secified column col. Returns the average 
# of that column of df.
def avg_val(df, col):
    avg_value=df[col].mean()
    return avg_value

avg_Energy_Supply_per_Capita = avg_val(final_data, 'Energy Supply per Capita')
avg_Energy_Supply_per_Capita

###############################################################
# Q5)
# takes the dataframe df and gives the value of col1 which has the max value of col2 
def max_val(df,col1, col2):
    max_data = str(df[col1][df[col2] == df[col2].max()][0])
    max_percent = float(df[col2].max())
    max_tup = (max_data , max_percent)
    return max_tup

max_renew = max_val(final_data,'Country', '% Renewable')
max_renew

###############################################################
# Q6)
# takes the dataframe df, calculates the ratio of columns col2 and col3 in that
# dataframe and finds the value in column col1 with max of that ratio. 
def ratio(df, col1, col2, col3):
    df['new_col'] = [float(x)/y if y !=0 else np.nan for (x,y) in \
      zip(df[col2], df[col3])]

    c = map(str,df[col1][df['new_col'] == \
                       max(df['new_col'])])[0]
    r = float(max(df['new_col']))
    return (c,r)

(count,ratio) = ratio(final_data, 'Country', 'Self-citations', 'Citations')
(count,ratio)

###############################################################
# Q7)
# Calculates an estimate of the population of countries from the ratio of 
# energy supply to the energy supply per capita.
flf = lambda x: "%.30f" % x
final_data['Population'] = [flf(float(x)/y) if y !=0 else np.nan for (x,y) in \
      zip(final_data['Energy Supply'], final_data['Energy Supply per Capita'])]

final_data['Population'].head()

################

# Takes a dataframe df and three of its column names and calculates the 3rd col1 in terms of 
# the sorted value of the ratio of col2 and col3.
def val_est(df, col1, col2, col3):
    df['Population'] = [float(x)/y if y !=0 else np.nan for (x,y) in \
      zip(df[col2], df[col3])]

    col1_3rd = df[[col1, 'Population']].sort_values(['Population'],\
                  ascending = False)[col1].iloc[2]
    return str(col1_3rd)

country_3rd = val_est(final_data, 'Country', 'Energy Supply', 'Energy Supply per Capita')
country_3rd

###############################################################
# Q8)
# Calculates the citable documents per capita by dividing the citable documents
# by the population.
final_data['Citable docs per Capita'] = [float(x)/y if y !=0 else np.nan for (x,y) in \
      zip(final_data['Citable documents'], final_data['Population'])]

# takes dataframe df and finds the correlation between columns col2 and col3
# in the df 
def correl(df, col1, col2):
    cor = df[col1].astype('float64')\
              .corr(df[col2].astype('float64')\
                    , method='pearson')
    return cor

correlation = correl(final_data, 'Citable docs per Capita', \
                     'Energy Supply per Capita')
correlation

#######################
# plots a scatter diagram of the energy supply per capita and citable docs
# per capita
plt.rc('figure', figsize=(20, 12))
plt.plot(final_data['Citable docs per Capita'], \
         final_data['Energy Supply per Capita'], 'o', markersize=20)
plt.xlim(0,0.0006)
plt.ylim(0,350)
plt.tick_params(labelsize=30)
plt.tick_params(labelsize=30, length=15, width = 2, pad = 10)
plt.xlabel('Citable docs per Capita', fontsize=35, labelpad=20)
plt.ylabel('Energy Supply per Capita', fontsize=35, labelpad=20)
plt.show()
###############################################################
# Q9)
# takes dataframe df and using the median of col1, creates a new column by 
# deciding whether the values of col1 are bigger or smaller than the median
# as an output it returns a series with values of the new column sorted 
# according to col3, with col2 as its index. 
def highval(df, col1, col2, col3):
    med = df[col1].median()
    df['Renew_level'] = [1 if x >= med else 0 for x in df[col1]]
    HR = df[[col2, col3, 'Renew_level']].sort_values([col3])
    HR = HR[[col2, 'Renew_level']]
    HR = HR.set_index(col2)
    return HR

HighRenew =  highval(final_data, '% Renewable', 'Country', 'Rank')
HighRenew

###############################################################
# Q10)
ContinentDict  = {'China':'Asia', 
                   'United States':'North America', 
                   'Japan':'Asia', 
                   'United Kingdom':'Europe', 
                   'Russian Federation':'Europe', 
                   'Canada':'North America', 
                   'Germany':'Europe', 
                   'India':'Asia',
                   'France':'Europe', 
                   'South Korea':'Asia', 
                   'Italy':'Europe', 
                   'Spain':'Europe', 
                   'Iran':'Asia',
                   'Australia':'Australia', 
                   'Brazil':'South America'}

cont_func = lambda x: ContinentDict[x]

# takes dataframe df, a key to be used for grouping, a col to be uses as the
# basis of statistical calculations. Returns a new dataframe contaning the 
# calculated statistics for the groups produced based on the key of df, with 
# the key as the index. The key however is a new key obtained from the input
# key by using the above dictionay as a map.
def new_df(dff, key, col):
    df = dff.copy()
    df.index = df.index.map(cont_func)
    df[key] = df.index
    stat = df.groupby(key).agg(['size', 'sum', 'mean', 'std'])[col]
    return stat

new_df(final_data, 'Continent', 'Population')

###############################################################
# Q11)
# creates 5 bins of equal size for the "% Renewable" column
m = final_data['% Renewable'].max()
n = final_data['% Renewable'].min()
final_data['% Renewable bins'] = \
['(%(1)f, %(2)f]' % {'1': n, '2':(n + (m-n)/5.0)} if x <= (n + (m-n)/5.0)\
 else '(%(1)f, %(2)f]' % {'1': (n + (m-n)/5.0), '2':2*(n + (m-n)/5.0)} if  \
    (x > (n + (m-n)/5.0) and x <= (n + 2*(m-n)/5.0))\
else '(%(1)f, %(2)f]' % {'1': 2*(n + (m-n)/5.0), '2':3*(n + (m-n)/5.0)} if \
    (x > (n + 2*(m-n)/5.0) and x <= (n + 3*(m-n)/5.0))\
else '(%(1)f, %(2)f]' % {'1': 3*(n + (m-n)/5.0), '2':4*(n + (m-n)/5.0)} if \
    (x > (n + 3*(m-n)/5.0) and x <= (n + 4*(m-n)/5.0))\
else '(%(1)f, %(2)f]' % {'1': 4*(n + (m-n)/5.0), '2':m} \
    for x in final_data['% Renewable']]

# Takes the dataframe df, and two columns col1 and col2 and groups df
# according to the two columns. The last column of the resulting data
# gives the number of countries in each group. This is why I didn't remove
# that column.   
def dou_gr(df,col1,col2):
    df[col1] = df.index.map(cont_func)
    dgr = df.sort('% Renewable').groupby([col1, col2]).agg(['size'])['Country']
    return dgr

double_grouping = dou_gr(final_data,'Continent', '% Renewable bins')
double_grouping

###############################################################
# Q12)
flf = lambda x: "%.30f" % x
final_data['Population'] = [flf(float(x)/y) if y !=0 else np.nan for (x,y) in \
      zip(final_data['Energy Supply'], final_data['Energy Supply per Capita'])]

# takes a numeric column (col_old) of a dataframe (df) and returns a new
# column (col_new) with those numbers as strings with thousands separators.
def pop_est(df, col_old, col_new):
    df[col_new] = df[col_old]
    j = 0
    for i in df[col_old]:
        st = list(str(i))
        ind = st.index('.')
        pos = ind-3
        while (pos > 0):
            st.insert(pos,',')
            pos -= 3
        st = ''.join(st)
        df[col_new].iloc[j] = st
        j  += 1
    return df[col_new]

PopEst = pop_est(final_data, 'Population', 'PopEst')
PopEst


