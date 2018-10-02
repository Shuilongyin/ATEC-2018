# -*- coding: utf-8 -*-
"""
Created on Mon May 21 14:31:03 2018

@author: evan
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

os.chdir('D:/ML_project\蚂蚁欺诈风险大赛_20180426/')

#画变量均值随时间变化的散点图

data_offline_avg = pd.read_csv('data/eda/data_offline_avg.csv')

data_all_date_avg = pd.read_csv('data/eda/data_all_date_avg.csv')

data_offline_good_std_mean_ratio = pd.read_csv('D:/ML_project/蚂蚁欺诈风险大赛_20180426/data/eda/data_offline_good_std_mean_ratio.csv',\
                                               header=None,names=['var','std_mean'],engine='python').\
                                               sort_values(by='std_mean',ascending=False)


data_offline_avg.loc[:,['f1']].plot()

for i in data_all_date_avg.columns[60:90]:
    data_all_date_avg.loc[:,[i]].plot()

#各标签均值数据数据随时间变化

data_offline_good_date_avg = pd.read_csv('data/eda/data_offline_good_date_avg.csv')
data_offline_bad_date_avg = pd.read_csv('data/eda/data_offline_bad_date_avg.csv')
data_offline_nolabel_date_avg = pd.read_csv('data/eda/data_offline_nolabel_date_avg.csv')

df_fi_0523 = pd.read_csv('D:\ML_project\蚂蚁欺诈风险大赛_20180426\data\df_fi_0523.csv',engine='python')

for i in data_all_date_avg.columns[20:36]:
    data_offline_good_date_avg.loc[:,[i]].merge(data_offline_bad_date_avg.loc[:,[i]],left_index=True,\
                                  right_index=True,suffixes =['_good','_bad'],how='left').merge\
                                  (data_offline_nolabel_date_avg.loc[:,[i]],\
                                   left_index=True,right_index=True, suffixes =['','_nolabel'],\
                                   how='left').merge\
                                  (data_all_date_avg.loc[:,[i]],\
                                   left_index=True,right_index=True, suffixes =['_nolabel','_all'],\
                                   how='left').plot()      
                                  
for i in df_fi_0523.head(10)['var']:
    data_offline_good_date_avg.loc[:,[i]].merge(data_offline_bad_date_avg.loc[:,[i]],left_index=True,\
                                  right_index=True,suffixes =['_good','_bad'],how='left').merge\
                                  (data_offline_nolabel_date_avg.loc[:,[i]],\
                                   left_index=True,right_index=True, suffixes =['','_nolabel'],\
                                   how='left').merge\
                                  (data_all_date_avg.loc[:,[i]],\
                                   left_index=True,right_index=True, suffixes =['_nolabel','_all'],\
                                   how='left').plot()


##查看各变量的缺失率随时间变化

data_all_na_avg = pd.read_csv('data/eda/data_all_na_avg.csv')

for i in data_all_date_avg.columns[6:9]:
    data_all_na_avg.loc[:,[i]].plot()  











