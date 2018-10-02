import pandas as pd
from dateutil.parser import parse
import numpy as np
import gc
import matplotlib.pyplot as plt
import os


def creatDtype():
    dtype = {'id':'object',
             'label':'int8',
             'date':'int64',
             'f1':'uint8',
             'f2': 'uint8',
             'f3': 'uint8',
             'f4': 'uint8',
             'f5': 'float32',
             'ndays':'uint8'
             }
    for i in range(20,298):
        dtype['f'+str(i)] = 'float32'
    for i in range(6,20):
        dtype['f'+str(i)] = 'uint8'
    return dtype

train_data = pd.read_csv('or_data\\atec_anti_fraud_train.csv',dtype=creatDtype())
test_data_a = pd.read_csv('or_data\\atec_anti_fraud_test_a.csv',dtype=creatDtype())
test_data_b = pd.read_csv('or_data\\atec_anti_fraud_test_b.csv',dtype=creatDtype())
data = train_data.append(test_data_a).append(test_data_b).reset_index(drop=True)
del train_data,test_data_a,test_data_b

data['ndays'] = data['date'].apply(lambda x:(parse(str(x))-parse(str(20170905))).days)
pre = [k for k in data.columns if k not in ['id', 'label', 'date', 'ndays']]

if not os.path.exists('var_distribution'):
    os.makedirs('var_distribution')

if not os.path.exists('var_miss_distribution'):
    os.makedirs('var_miss_distribution')

##根据缺失分布的时间趋势，了解数据,可以知道这份数据来源是几个表
for p in pre:
    a = pd.DataFrame(data.groupby('ndays')[p].apply(lambda x:sum(pd.isnull(x)))/data.groupby('ndays')['ndays'].count()).reset_index()
    a.columns = ['ndays',p]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(a['ndays'],a[p])
    plt.axvline(30, color='r')
    plt.axvline(40, color='r')
    plt.axvline(61, color='r')
    plt.axvline(122, color='r')
    plt.axvline(153, color='r')
    plt.xlabel('ndays')
    plt.ylabel('miss_rate_'+p)
    plt.title('miss_rate_'+p)
    plt.savefig('var_miss_distribution\\'+p+'_miss_rate.png')


##查看数据具体值的分布，可以看出具体数据的分布变化
for p in pre:
    a = pd.DataFrame(data.groupby('ndays')[p].mean()).reset_index()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(a['ndays'],a[p])
    plt.axvline(30, color='r')
    plt.axvline(40, color='r')
    plt.axvline(61, color='r')
    plt.axvline(122, color='r')
    plt.axvline(153, color='r')
    plt.xlabel('ndays')
    plt.ylabel('mean_of_'+p)
    plt.title('distribution of '+p)
    plt.savefig('var_distribution\\'+p+'_mean_dis')


##还需要看一个每个特征的风险趋势（建模工具），把灰黑合并




'''
特征选择的基本原则，之前A选择的原则是根据数值的分布，删除随着时间趋势 平缓了的，以及分布随时间倾向是与测试相反的，
实际要做的更多，首先要根据特征缺失的维度，确定有几个表的数据，每个表的特征分别处理，若缺失值前后变化非常大，需要看下特征的风险，缺失是否有比较高的风险，若除了缺失，其他值没有明显区分，考虑删除此特征，若其他值有区分，需要单独处理这部分数据
样本选择的方法：样本选择的方法，目前没有找到最优，目前线上最优的是只用前40天数据建模
'''






