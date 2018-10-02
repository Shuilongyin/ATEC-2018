# -*- coding: utf-8 -*-
"""
Created on Fri May  4 17:29:12 2018

@author: evan

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
data_online = pd.read_csv('data/test_data.csv')

no_predictors = ['id','label','date']
predictors = [i for i in data_offline.columns if i not in no_predictors]

#查看数据集大小
data_offline.shape #(994731, 300)
data_online.shape #(491668, 299)

#查看发生日期分布
data_offline.groupby(by='date',as_index=False)['id'].agg({'cnt':'count'}).sort_values(by='date') #从20170905-20171105，
data_online.groupby(by='date',as_index=False)['id'].agg({'cnt':'count'}).sort_values(by='date') #从20180105-20180205

#查看线下标签分布
data_offline.groupby(by='label',as_index=False)['id'].agg({'cnt':'count'}).sort_values(by='label')

# label     cnt
# 0     -1    4725
# 1      0  977884
# 2      1   12122

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

# #简单分箱，计算iv #数据太大，暂不可行
# from woe_iv import Woe_iv

# no_predictors = ['id','label','date']
# predictors = [i for i in data_offline.columns if i not in no_predictors]

# data_offline_label_filter = data_offline.loc[data_offline.loc[:,'label'].isin([0,1]),:]

# wi = Woe_iv(data_offline_label_filter[predictors+['label']],dep='label',disnums=10)
# wi.woe_iv_vars()
# iv_df = pd.DataFrame(pd.Series(wi.iv_dict)).reset_index().rename(columns={'index':'var',0:'iv'}).sort_values(by='iv',ascending=False)

'''
#训练一个base-model
'''

import os 
import pandas as pd
import numpy as np
import sys
import csv
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from sklearn.metrics import roc_curve, auc
import copy
import warnings
import pickle
warnings.filterwarnings("ignore")

def ant_score(y_true, predict_proba):
    '''
    y_true: numpy.ndarray,不能是带索引的series
    '''
    fpr, tpr, thresholds = roc_curve(y_true, predict_proba, pos_label=1) 
    score=0.4*tpr[np.where(fpr>=0.001)[0][0]]+0.3*tpr[np.where(fpr>=0.005)[0][0]]+0.3*tpr[np.where(fpr>=0.01)[0][0]] 
    return score 

def modelfit(alg, dtrain, predictors,dtest,useTrainCV=True, cv_folds=5, early_stopping_rounds=100):
    global df_fi
    global cvresult
    global dtest_predprob
    global test_prob
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[dep].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[dep],eval_metric='auc')
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtest_predprob = alg.predict_proba(dtest[predictors])[:,1]
    ks_test = ant_score(dtest[dep],dtest_predprob)
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    ks_train = ant_score(dtrain[dep],dtrain_predprob)
    false_positive_rate, recall, thresholds = roc_curve(dtest[dep],dtest_predprob)
    roc_auc = auc(false_positive_rate, recall)
    test_prob = pd.DataFrame([dtest[dep].values,dtest_predprob]).T
    print ("\nModel Report")
    print ("n_estimators: %d" % cvresult.shape[0])
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain[dep].values, dtrain_predictions))
    print ("AUC Score Mean(Train): %f" %cvresult.loc[cvresult.shape[0]-1,'train-auc-mean'] )
    print ("AUC Score Std (Train): %f" %cvresult.loc[cvresult.shape[0]-1,'train-auc-std'] )
    print ("AUC Score Mean(Test): %f" %cvresult.loc[cvresult.shape[0]-1,'test-auc-mean'] )
    print ("AUC Score Std (Test): %f" %cvresult.loc[cvresult.shape[0]-1,'test-auc-std'] )
    print ('ks (Train): %f'%ks_train)
    print ('ks (Test): %f'%ks_test)
    print ('AUC (dtest):%f'%roc_auc)
    featureimportance=pd.DataFrame(alg.feature_importances_)
    feat_names =pd.DataFrame(predictors)
    df_fi=pd.merge(feat_names,featureimportance,left_index=True,right_index=True).sort_values(by=['0_y'],ascending=False)
    df_fi.rename(columns={'0_y':'FeatureImportanceScore','0_x':'var'}, inplace = True)
    df_fi = df_fi.merge(it_dict,on='var',how='left')
    return [roc_auc,ks_test]
    #df_fi.plot('var','FeatureImportanceScore',kind='bar', title='Feature Importances')
    #plt.ylabel('Feature Importance Score')
    
#将最后10天的数据作为线下test
no_predictors = ['id','label','date']
predictors = [i for i in data_offline.columns if i not in no_predictors]

data_offline_label_filter = data_offline.loc[data_offline.loc[:,'label'].isin([0,1]),:]

data_offline_label_filter_train = data_offline_label_filter.loc[data_offline_label_filter.loc[:,'date']<=20171026,:]
data_offline_label_filter_test = data_offline_label_filter.loc[data_offline_label_filter.loc[:,'date']>20171026,:]


#xgb模型(很慢)
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=2000,
 max_depth=5,
 min_child_weight=1,
 gamma=0.1,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=8,
 scale_pos_weight=1,
 seed=1,
 reg_alpha=0.1)
 
dep='label'

modelfit(xgb1, data_offline_label_filter_train, predictors,data_offline_label_filter_test)


#rf模型（十分钟）
data_offline_label_filter_train_fillna = data_offline_label_filter_train.fillna(0)
data_offline_label_filter_test_fillna = data_offline_label_filter_test.fillna(0)

from sklearn.ensemble import RandomForestClassifier
rf2 = RandomForestClassifier(n_estimators =1000,n_jobs =8,random_state =3,max_depth =5)
rf2.fit(data_offline_label_filter_train_fillna[predictors],data_offline_label_filter_train_fillna[dep])
test_proba = rf2.predict_proba(data_offline_label_filter_test_fillna[predictors])[:,1]
false_positive_rate, recall, thresholds = roc_curve(data_offline_label_filter_test_fillna[dep],test_proba)
print(auc(false_positive_rate, recall))
print (ant_score(data_offline_label_filter_test[dep],test_proba)) #0.4

data_online_fillna = data_online.fillna(0)
data_online_proba = rf2.predict_proba(data_online_fillna[predictors])[:,1]

data_online_proba_df = pd.DataFrame(pd.Series(data_online_proba)).rename(columns={0:'score'})
data_online_proba_df['id'] = data_online['id'].tolist()
data_online_proba_df.loc[:,['id','score']].to_csv('data/data_online_proba_df(base_model_0520).csv',encoding='utf8',index=False) #0.234


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

data_all_date_avg_std = data_all_date_avg.std()
data_all_date_avg_mean = data_all_date_avg.mean()
data_all_date_avg_std_mean_ratio = data_all_date_avg_std/data_all_date_avg_mean
pd.DataFrame(data_all_date_avg_std_mean_ratio).to_csv('data/data_all_date_avg_std_mean_ratio.csv',encoding='gbk')

#分标签数据(将online的均值放进label=0的地方)
data_offline_good = data_offline.loc[data_offline.loc[:,'label']==0,:]
data_offline_good = pd.concat([data_offline_good,data_online])
data_offline_good_date_avg = data_offline_good.groupby(by='date',as_index=False)[predictors+['label']].mean()
data_offline_good_date_avg.to_csv('data/data_offline_good_date_avg.csv',encoding='gbk',index=False)
data_offline_good_std_mean_ratio = data_offline_good_date_avg.std()/data_offline_good_date_avg.mean()
pd.DataFrame(data_offline_good_std_mean_ratio).to_csv('data/data_offline_good_std_mean_ratio.csv',encoding='gbk')

data_offline_bad = data_offline.loc[data_offline.loc[:,'label']==1,:]
data_offline_bad_date_avg = data_offline_bad.groupby(by='date',as_index=False)[predictors+['label']].mean()
data_offline_bad_date_avg.to_csv('data/data_offline_bad_date_avg.csv',encoding='gbk',index=False)

data_offline_nolabel = data_offline.loc[data_offline.loc[:,'label']==-1,:]
data_offline_nolabel_date_avg = data_offline_nolabel.groupby(by='date',as_index=False)[predictors+['label']].mean()
data_offline_nolabel_date_avg.to_csv('data/data_offline_nolabel_date_avg.csv',encoding='gbk',index=False)



#各特征中位数随时间的分布
data_all = pd.concat([data_offline,data_online])

data_all_date_median = data_all.groupby(by='date',as_index=False)[predictors+['label']].median()
data_all_date_median.to_csv('data/data_all_date_median.csv',encoding='gbk',index=False)
#分标签数据(将online的中位数放进label=0的地方)
data_offline_good = data_offline.loc[data_offline.loc[:,'label']==0,:]
data_offline_good = pd.concat([data_offline_good,data_online])
data_offline_good_date_median = data_offline_good.groupby(by='date',as_index=False)[predictors+['label']].median()
data_offline_good_date_median.to_csv('data/data_offline_good_date_median.csv',encoding='gbk',index=False)


data_offline_bad = data_offline.loc[data_offline.loc[:,'label']==1,:]
data_offline_bad_date_median = data_offline_bad.groupby(by='date',as_index=False)[predictors+['label']].median()
data_offline_bad_date_median.to_csv('data/data_offline_bad_date_median.csv',encoding='gbk',index=False)

data_offline_nolabel = data_offline.loc[data_offline.loc[:,'label']==-1,:]
data_offline_nolabel_date_median = data_offline_nolabel.groupby(by='date',as_index=False)[predictors+['label']].median()
data_offline_nolabel_date_median.to_csv('data/data_offline_nolabel_date_median.csv',encoding='gbk',index=False)

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

#查看某些变量间的相关性
date_sorted = sorted(list(set(data_offline['date'])))
na_col = ['f'+str(i) for i in  range(64,155)]
train_col = data_offline.loc[data_offline.loc[:,'date'].isin(date_sorted[:46]),:]
df_corr_na_col = train_col.fillna(0).loc[:,na_col].corr().applymap(lambda x: abs(x))
df_corr_na_col.to_csv('data/df_corr_na_col.csv',encoding='gbk')






