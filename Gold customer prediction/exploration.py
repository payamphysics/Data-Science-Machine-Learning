# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 12:38:10 2017

@author: Payam
"""

#plt.scatter(merged_data_all['max_to_std'], merged_data_all['gold'])

plt.figure(figsize=(20, 20))
plt.scatter(merged_data_all['spend_med_total'], merged_data_all['gold'])

plt.figure(figsize=(20, 20))
plt.hist(merged_data_all['income'][merged_data_all['income'].notnull()][merged_data_all['gold'] == 0], bins= 400)
plt.hist(merged_data_all['income'][merged_data_all['income'].notnull()][merged_data_all['gold'] == 1], bins= 400)
plt.show()

plt.figure(figsize=(20, 20))
plt.hist(merged_data_all['age'][merged_data_all['gold'] == 0], bins= 100, range=(0,100))
plt.hist(merged_data_all['age'][merged_data_all['gold'] == 1], bins= 100, range=(0,100))
plt.show()

plt.figure(figsize=(20, 20))
plt.hist(merged_data_all['sex'][merged_data_all['gold'] == 1], bins= 100, range=(0,4))
plt.hist(merged_data_all['sex'][merged_data_all['gold'] == 0], bins= 100, range=(0,4))
plt.show()

plt.figure(figsize=(20, 20))
plt.hist(merged_data_all['count_total'][merged_data_all['gold'] == 0], bins= 100, range=(0,100))
plt.hist(merged_data_all['count_total'][merged_data_all['gold'] == 1], bins= 100, range=(0,100))

plt.figure(figsize=(20, 20))
plt.hist(merged_data_all['spend_max_total'][merged_data_all['gold'] == 0], bins= 500, range=(0,1000))
plt.hist(merged_data_all['spend_max_total'][merged_data_all['gold'] == 1], bins= 500, range=(0,1000))

plt.figure(figsize=(20, 20))
plt.hist(merged_data_all['spend_std_total'][merged_data_all['gold'] == 0], bins= 500, range=(0,1000))
plt.hist(merged_data_all['spend_std_total'][merged_data_all['gold'] == 1], bins= 500, range=(0,1000))


plt.figure(figsize=(20, 20))
plt.hist(merged_data_all['max_to_std'][merged_data_all['gold'] == 0], bins= 100, range=(0,5))
plt.hist(merged_data_all['max_to_std'][merged_data_all['gold'] == 1], bins= 100, range=(0,5))


plt.scatter(merged_data_all['spend_total'],merged_data_all['gold'])
plt.scatter(merged_data_all['count_total'],merged_data_all['gold'])
plt.scatter(merged_data_all['age'],merged_data_all['gold'])
plt.scatter(merged_data_all['sex'],merged_data_all['gold'])
plt.scatter(merged_data_all['credit_flag'],merged_data_all['gold'])
plt.scatter(merged_data_all['count_g'],merged_data_all['gold'])
plt.scatter(merged_data_all['count_s'],merged_data_all['gold'])
plt.scatter(merged_data_all['count_o'],merged_data_all['gold'])
plt.scatter(merged_data_all['spend_g'],merged_data_all['gold'])
plt.scatter(merged_data_all['spend_s'],merged_data_all['gold'])
plt.scatter(merged_data_all['spend_o'],merged_data_all['gold'])
plt.scatter(merged_data_all['spend_avg_g'],merged_data_all['gold'])
plt.scatter(merged_data_all['spend_avg_s'],merged_data_all['gold'])
plt.scatter(merged_data_all['spend_avg_o'],merged_data_all['gold'])
plt.scatter(merged_data_all['spend_avg_total'],merged_data_all['gold'])
plt.scatter(merged_data_all['spend_std_total'],merged_data_all['gold'])
plt.scatter(merged_data_all['spend_max_total'],merged_data_all['gold'])

plt.figure(figsize=(20, 20))
plt.hist(merged_data_all['spend_total'][merged_data_all['gold'] == 0], bins= 100, range=(0,2000))
plt.hist(merged_data_all['spend_total'][merged_data_all['gold'] == 1], bins= 100, range=(0,2000))
plt.show()

plt.figure(figsize=(20, 20))
plt.hist(merged_data_all['age'][merged_data_all['gold'] == 0], bins= 100, range=(0,100))
plt.hist(merged_data_all['age'][merged_data_all['gold'] == 1], bins= 100, range=(0,100))
plt.show()

plt.figure(figsize=(20, 20))
plt.hist(merged_data_all['sex'][merged_data_all['gold'] == 1], bins= 100, range=(0,4))
plt.hist(merged_data_all['sex'][merged_data_all['gold'] == 0], bins= 100, range=(0,4))
plt.show()




plt.figure(figsize=(20, 20))
plt.hist(merged_data_all['spend_max_total'][merged_data_all['gold'] == 0], bins= 100, range=(0,2000))
plt.hist(merged_data_all['spend_max_total'][merged_data_all['gold'] == 1], bins= 100, range=(0,2000))
plt.show()

plt.figure(figsize=(20, 20))
plt.hist(merged_data_all['spend_max_total'][merged_data_all['gold'] == 0], bins= 50, range=(0,250))
plt.hist(merged_data_all['spend_max_total'][merged_data_all['gold'] == 1], bins= 50, range=(0,250))
plt.show()

plt.figure(figsize=(20, 20))
plt.hist(merged_data_all['spend_max_total'][merged_data_all['gold'] == 1], bins= 50, range=(250,500))
plt.hist(merged_data_all['spend_max_total'][merged_data_all['gold'] == 0], bins= 50, range=(250,500))
plt.show()

plt.figure(figsize=(20, 20))
plt.hist(merged_data_all['spend_max_total'][merged_data_all['gold'] == 1], bins= 50, range=(500, 1000))
plt.hist(merged_data_all['spend_max_total'][merged_data_all['gold'] == 0], bins= 50, range=(500, 1000))
plt.show()

plt.figure(figsize=(20, 20))
plt.hist(merged_data_all['spend_max_total'][merged_data_all['gold'] == 1], bins= 50, range=(1000, 2000))
plt.hist(merged_data_all['spend_max_total'][merged_data_all['gold'] == 0], bins= 50, range=(1000, 2000))
plt.show()




plt.figure(figsize=(20, 20))
plt.hist(merged_data_all['spend_avg_total'][merged_data_all['gold'] == 0], bins= 100, range=(0,2000))
plt.hist(merged_data_all['spend_avg_total'][merged_data_all['gold'] == 1], bins= 100, range=(0,2000))
plt.show()

plt.figure(figsize=(20, 20))
plt.hist(merged_data_all['spend_avg_total'][merged_data_all['gold'] == 0], bins= 50, range=(0,250))
plt.hist(merged_data_all['spend_avg_total'][merged_data_all['gold'] == 1], bins= 50, range=(0,250))
plt.show()

plt.figure(figsize=(20, 20))
plt.hist(merged_data_all['spend_avg_total'][merged_data_all['gold'] == 1], bins= 50, range=(250,500))
plt.hist(merged_data_all['spend_avg_total'][merged_data_all['gold'] == 0], bins= 50, range=(250,500))
plt.show()

plt.figure(figsize=(20, 20))
plt.hist(merged_data_all['spend_avg_total'][merged_data_all['gold'] == 1], bins= 50, range=(500, 1000))
plt.hist(merged_data_all['spend_avg_total'][merged_data_all['gold'] == 0], bins= 50, range=(500, 1000))
plt.show()

plt.figure(figsize=(20, 20))
plt.hist(merged_data_all['spend_avg_total'][merged_data_all['gold'] == 1], bins= 50, range=(1000, 2000))
plt.hist(merged_data_all['spend_avg_total'][merged_data_all['gold'] == 0], bins= 50, range=(1000, 2000))
plt.show()




plt.figure(figsize=(20, 20))
plt.hist(merged_data_all['spend_std_total'][merged_data_all['gold'] == 1], bins= 300, range=(0,250))
plt.hist(merged_data_all['spend_std_total'][merged_data_all['gold'] == 0], bins= 300, range=(0,250))
plt.show()

plt.figure(figsize=(20, 20))
plt.hist(merged_data_all['spend_std_total'][merged_data_all['gold'] == 1], bins= 100, range=(250,500))
plt.hist(merged_data_all['spend_std_total'][merged_data_all['gold'] == 0], bins= 100, range=(250,500))
plt.show()

plt.figure(figsize=(20, 20))
plt.hist(merged_data_all['spend_std_total'][merged_data_all['gold'] == 1], bins= 100, range=(500, 1000))
plt.hist(merged_data_all['spend_std_total'][merged_data_all['gold'] == 0], bins= 100, range=(500, 1000))
plt.show()

plt.figure(figsize=(20, 20))
plt.hist(merged_data_all['spend_std_total'][merged_data_all['gold'] == 1], bins= 100, range=(1000, 2000))
plt.hist(merged_data_all['spend_std_total'][merged_data_all['gold'] == 0], bins= 100, range=(1000, 2000))
plt.show()

