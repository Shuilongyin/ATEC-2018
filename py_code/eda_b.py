# -*- coding: utf-8 -*-
"""
Created on Fri May  4 17:29:12 2018

@author: evan

B榜
"""

import os 
import pandas as pd
import numpy as np
import sys
import csv
import copy

os.chdir('/mnt/Data/xiongwenwen/ant_risk/')
sys.path.append('/mnt/Data/xiongwenwen/py_code')

#读取线下训练集和线上测试集
data_offline = pd.read_csv('data/train_data.csv')
data_online = pd.read_csv('data/test_b.csv')
data_online_a = pd.read_csv('data/test_data.csv')

no_predictors = ['id','label','date']
predictors = [i for i in data_offline.columns if i not in no_predictors]

#查看数据集大小
data_offline.shape #(994731, 300)
data_online.shape #(500538, 299)

#查看发生日期分布
data_offline.groupby(by='date',as_index=False)['id'].agg({'cnt':'count'}).sort_values(by='date') #从20170905-20171105，
data_online.groupby(by='date',as_index=False)['id'].agg({'cnt':'count'}).sort_values(by='date') #从20180206-20180306

#查看线下标签分布
data_offline.groupby(by='label',as_index=False)['id'].agg({'cnt':'count'}).sort_values(by='label')

# label     cnt
# 0     -1    4725
# 1      0  977884
# 2      1   12122

 # 0    0.983064
 # 1    0.012186
# -1    0.004750

#查看日期下的标签分布
data_offline_date_group = data_offline.groupby(by='date',as_index=False)['id'].agg({'cnt':'count'}).sort_values(by='date')
data_offline_data_bad_group = data_offline.loc[data_offline.loc[:,'label']==1,:].groupby(by='date',as_index=False)['id'].agg({'bad_cnt':'count'}).sort_values(by='date')
data_offline_date_group_merge = data_offline_date_group.merge(data_offline_data_bad_group,how='left')
data_offline_date_group_merge['bad_ratio'] = data_offline_date_group_merge['bad_cnt']/data_offline_date_group_merge['cnt']
data_offline_date_group_merge.to_csv('data/data_offline_date_group_merge.csv',encoding='gbk',index=False)

data_offline_data_reject_group = data_offline.loc[data_offline.loc[:,'label']==-1,:].groupby(by='date',as_index=False)['id'].agg({'re_cnt':'count'}).sort_values(by='date')
data_offline_date_group_merge_re = data_offline_date_group.merge(data_offline_data_reject_group,how='left')
data_offline_date_group_merge_re['reject_ratio'] = data_offline_date_group_merge_re['re_cnt']/data_offline_date_group_merge_re['cnt']
data_offline_date_group_merge_re.to_csv('data/data_offline_date_group_merge_re.csv',encoding='gbk',index=False)

#查看各特征的分布
data_offline_desc = data_offline.describe()
data_offline_desc.to_csv('data/data_offline_desc.csv',encoding='gbk') #某些字段的缺失率一样，貌似来自同一数据集

data_online_desc = data_online.describe()
data_online_desc.to_csv('data/data_online_desc.csv',encoding='gbk') #某些字段的缺失率一样，貌似来自同一数据集

'''
各特征的分布
'''
#各特征均值随时间的分布

#线下数据
data_offline_date_avg = data_offline.groupby(by='date',as_index=False)[predictors+['label']].mean()
data_offline_date_avg.to_csv('data/data_offline_date_avg.csv',encoding='gbk',index=False)

data_offline_date_avg_std = data_offline_date_avg.std()
data_offline_date_avg_mean = data_offline_date_avg.mean()
data_offline_date_avg_std_mean_ratio = data_offline_date_avg_std/data_offline_date_avg_mean
pd.DataFrame(data_offline_date_avg_std_mean_ratio).to_csv('data/data_offline_date_avg_std_mean_ratio.csv',encoding='gbk')

#全体数据
data_all = pd.concat([data_offline,data_online])

data_all_date_avg = data_all.groupby(by='date',as_index=False)[predictors+['label']].mean()
data_all_date_avg.to_csv('data/data_all_date_avg.csv',encoding='gbk',index=False)


#分标签数据
data_offline_good = data_offline.loc[data_offline.loc[:,'label']==0,:]
data_offline_good = pd.concat([data_offline_good,data_online])
data_offline_good_date_avg = data_offline_good.groupby(by='date',as_index=False)[predictors+['label']].mean()
data_offline_good_date_avg.to_csv('data/data_offline_good_date_avg.csv',encoding='gbk',index=False)
#data_offline_good_std_mean_ratio = data_offline_good_date_avg.std()/data_offline_good_date_avg.mean()
#pd.DataFrame(data_offline_good_std_mean_ratio).to_csv('data/data_offline_good_std_mean_ratio.csv',encoding='gbk')

data_offline_bad = data_offline.loc[data_offline.loc[:,'label']==1,:]
data_offline_bad_date_avg = data_offline_bad.groupby(by='date',as_index=False)[predictors+['label']].mean()
data_offline_bad_date_avg.to_csv('data/data_offline_bad_date_avg.csv',encoding='gbk',index=False)

data_offline_nolabel = data_offline.loc[data_offline.loc[:,'label']==-1,:]
data_offline_nolabel_date_avg = data_offline_nolabel.groupby(by='date',as_index=False)[predictors+['label']].mean()
data_offline_nolabel_date_avg.to_csv('data/data_offline_nolabel_date_avg.csv',encoding='gbk',index=False)

'''
各特征随时间的缺失率
'''
data_offline_na = pd.isnull(data_offline[predictors])
data_online_na = pd.isnull(data_online[predictors])
data_offline_na['date'] = data_offline['date']
data_online_na['date'] = data_online['date']

data_all_na = pd.concat([data_offline_na,data_online_na])

data_all_na_avg = data_all_na.groupby(by='date',as_index=False)[predictors].mean()
data_all_na_avg.to_csv('data/data_all_na_avg.csv',encoding='gbk',index=False)


'''
查看单变量情况
'''
data_offline['f28'].value_counts()

# 1.0     391016
# 0.0     386098
# 2.0       6104
# 3.0        368
# 4.0         60
# 5.0         27
# 6.0         10
# 7.0          5
# 10.0         2
# 12.0         1
# 13.0         1
# 11.0         1
# 8.0          1
# 9.0          1

data_offline.loc[data_offline['f28']>1,'label'].value_counts()

# 0    3615
# 1    2809
# -1     157

data_offline['f20'].value_counts()

# 1.0     188934
# 0.0     184992
# 31.0    110299
# 32.0     96169
# 30.0     18209
# 2.0       9236
# 17.0      7505
# 25.0      7309
# 7.0       7248
# 24.0      7238
# 5.0       7148
# 12.0      7107
# 22.0      7091
# 4.0       7057
# 16.0      7051
# 13.0      7004
# 21.0      6848
# 3.0       6720
# 9.0       6643
# 14.0      6630
# 23.0      6594
# 20.0      6389
# 10.0      6377
# 27.0      6364
# 8.0       6296
# 15.0      6268
# 28.0      6240
# 11.0      6132
# 19.0      6094
# 18.0      6090
# 6.0       6082
# 29.0      5998
# 26.0      5921

data_offline.loc[data_offline['f20']>1,'label'].value_counts()

data_offline.loc[data_offline['f24']>1,'label'].value_counts()

data_offline.loc[data_offline['f32']>1,'label'].value_counts()


