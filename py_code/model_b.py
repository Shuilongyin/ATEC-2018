import os 
import pandas as pd
import numpy as np
import sys
import csv
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from sklearn.metrics import roc_curve, auc
import copy
import warnings
import pickle
import decimal
warnings.filterwarnings("ignore")
import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

os.chdir('/mnt/Data/xiongwenwen/ant_risk/')
sys.path.append('/mnt/Data/xiongwenwen/py_code')

def ant_score(y_true, predict_proba):
    '''
    y_true: numpy.ndarray,不能是带索引的series
    '''
    fpr, tpr, thresholds = roc_curve(y_true, predict_proba, pos_label=1)
    score=0.4*tpr[np.where(fpr>=0.001)[0][0]]+0.3*tpr[np.where(fpr>=0.005)[0][0]]+0.3*tpr[np.where(fpr>=0.01)[0][0]]
    return score

def save_obj(obj, name ):
    with open('pkl/pkl_'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('pkl/pkl_' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def model_predict(dtrain,dtest,predictors,dep,donline):
    xgb1 = XGBClassifier(
     learning_rate =0.05,
     n_estimators=3000,
     max_depth=6,
     min_child_weight=1,
     gamma=0.1,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'binary:logistic',
     n_jobs=16,
     scale_pos_weight=1,
     seed=1,
     reg_alpha=0.5,
     reg_lambda =10,
     silent=False)
    xgb1.fit(dtrain[predictors],dtrain[dep],eval_set=[(dtrain[predictors],dtrain[dep]),(dtest[predictors],dtest[dep])],eval_metric='auc',early_stopping_rounds=100)
    featureimportance=pd.DataFrame(xgb1.feature_importances_)
    featureimportance['var'] = predictors
    df_fi = featureimportance.rename(columns={0:'fi'}).sort_values(by='fi',ascending=False)
    dtest_predprob = xgb1.predict_proba(test_data[predictors],ntree_limit=xgb1.best_iteration)[:,1]
    false_positive_rate, recall, thresholds = roc_curve(test_data[dep],dtest_predprob)
    roc_auc = auc(false_positive_rate, recall)
    at_score = ant_score(test_data[dep],dtest_predprob)
    print ('at_score：',at_score)
    donline_predprob = xgb1.predict_proba(donline[predictors],ntree_limit=xgb1.best_iteration)[:,1]
    donline['score'] = donline_predprob
    return donline.loc[:,['id','score']],df_fi,dtest_predprob,[roc_auc,at_score]

def model_cv(data,dep,predictors):
    score = {}
    for i in range(0,53,10):
        test_date = date_sorted[i:i+10]
        train_date = [i for i in date_sorted if i not in test_date]
        test_data = data.loc[data.loc[:,'date'].isin(test_date),:]
        train_data = data.loc[data.loc[:,'date'].isin(train_date),:]
        xgb1 = XGBClassifier(
         learning_rate =0.05,
         n_estimators=3000,
         max_depth=6,
         min_child_weight=1,
         gamma=0.1,
         subsample=0.8,
         colsample_bytree=0.8,
         objective= 'binary:logistic',
         n_jobs=16,
         scale_pos_weight=1,
         seed=1,
         reg_alpha=0.5,
         reg_lambda =10,
         silent=False)
        xgb1.fit(train_data[predictors],train_data[dep],eval_set=[(train_data[predictors],train_data[dep]),(test_data[predictors],test_data[dep])],eval_metric='auc',early_stopping_rounds=20)
        dtest_predprob = xgb1.predict_proba(test_data[predictors])[:,1]
        false_positive_rate, recall, thresholds = roc_curve(test_data[dep],dtest_predprob)
        roc_auc = auc(false_positive_rate, recall)
        at_score = ant_score(test_data[dep],dtest_predprob)
        print ('at_score：',at_score)
        score[i] = [roc_auc,at_score]
    return score

def corr_filter(df_corr,corr_thresh):
    df_corr_trans = df_corr.applymap(lambda x:0 if x<0.85 else x)
    vars = df_corr.columns.tolist()
    vars_corr_sum = df_corr_trans.sum().to_dict()
    vars_corr_drop = []
    for var in vars:
        #print (var)
        var_list = df_corr.loc[df_corr[var]>=corr_thresh,[var]].index.tolist()
        if len(var_list)>1:
            var_corr_list = []
            for vv in var_list:
                var_corr_list.append([vv,vars_corr_sum[vv]])
            var_corr_list_sorted = sorted(var_corr_list,key=lambda x:x[1],reverse=True)
            if var_corr_list_sorted[0][0]==var:
                vars_corr_drop.append(var)
            else:
                var_sorted = [i[0] for i in var_corr_list_sorted]
                vars_corr_drop.extend(var_sorted[:var_sorted.index(var)])
    return set(vars_corr_drop)

def avg_diff(data,coll):
    data_col_avg = data.groupby(by='date',as_index=False)[coll].mean()
    data_col_avg['date'] = data_col_avg['date'].apply(lambda x:int(str(pd.to_datetime(str(x))+datetime.timedelta(days=1))[:10].replace('-','')))
    data_col_merge = data[['id','date']+coll].merge(data_col_avg,how='left',on='date',suffixes =('','_avg'))
    data_col_merge.loc[data_col_merge.loc[:,'date']==(data['date'].min()),[i+'_avg' for i in coll]] = data_col_avg.loc[data_col_avg.loc[:,'date']==(data_col_avg['date'].min()),coll].values
    for cc in coll:
        data_col_merge[cc+'_avg_diff'] = data_col_merge[cc]-data_col_merge[cc+'_avg']
    return data_col_merge[['id']+[i+'_avg_diff' for i in coll]]

def med_diff(data,coll):
    data_col_avg = data.groupby(by='date',as_index=False)[coll].median()
    data_col_avg['date'] = data_col_avg['date'].apply(lambda x:int(str(pd.to_datetime(str(x))+datetime.timedelta(days=1))[:10].replace('-','')))
    data_col_merge = data[['id','date']+coll].merge(data_col_avg,how='left',on='date',suffixes =('','_median'))
    data_col_merge.loc[data_col_merge.loc[:,'date']==(data['date'].min()),[i+'_median' for i in coll]] = data_col_avg.loc[data_col_avg.loc[:,'date']==(data_col_avg['date'].min()),coll].values
    for cc in coll:
        data_col_merge[cc+'_median_diff'] = data_col_merge[cc]-data_col_merge[cc+'_median']
    return data_col_merge[['id']+[i+'_median_diff' for i in coll]]

def model_na_train(data_offline_filter,data_online,col,predictors):
    data_offline_filter_col_nona = data_offline_filter.loc[pd.notnull(data_offline_filter.loc[:,col]),:]
    data_offline_filter_col_na = data_offline_filter.loc[pd.isnull(data_offline_filter.loc[:,col]),:]
    data_online_col_na =  data_online.loc[pd.isnull( data_online.loc[:,col]),:]
    data_online_col_nona =  data_online.loc[pd.notnull( data_online.loc[:,col]),:]
    k= pd.qcut(data_offline_filter_col_nona[col].tolist()+[data_offline_filter_col_nona[col].min()-1], 10, retbins=True, labels=False,duplicates ='drop')
    cutoffs = k[1]
    data_offline_filter_col_nona[col+'_dis'] = np.digitize(data_offline_filter_col_nona[col], cutoffs, right=True)
    data_online_col_nona[col+'_dis'] = np.digitize(data_online_col_nona[col], cutoffs, right=True)
    dep = col+'_dis'
    train_col = data_offline_filter_col_nona.loc[data_offline_filter_col_nona.loc[:,'date'].isin(date_sorted[:40]),:]
    valid_col = data_offline_filter_col_nona.loc[data_offline_filter_col_nona.loc[:,'date'].isin(date_sorted[40:46]),:]
    xgb1 = XGBClassifier(
     learning_rate =0.05,
     n_estimators=3000,
     max_depth=6,
     min_child_weight=1,
     gamma=0.1,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'binary:logistic',
     n_jobs=16,
     scale_pos_weight=1,
     seed=1,
     reg_alpha=0.5,
     reg_lambda =10,
     silent=False)
    xgb1.fit(train_col[predictors],train_col[dep],eval_set=[(train_col[predictors],train_col[dep]),(valid_col[predictors],valid_col[dep])],eval_metric='mlogloss',early_stopping_rounds=10)
    save_obj(xgb1,col+'_xgb') #保存模型
    save_obj(cutoffs,col+'_cut') #保存切点
    data_offline_filter_col_na_pred = xgb1.predict(data_offline_filter_col_na[predictors])
    data_online_col_na_pred = xgb1.predict(data_online_col_na[predictors])
    data_offline_filter_col_na[col+'_dis'] = data_offline_filter_col_na_pred
    data_online_col_na[col+'_dis'] = data_online_col_na_pred    
    return pd.concat([data_offline_filter_col_nona.loc[:,['id',col+'_dis']],data_offline_filter_col_na.loc[:,['id',col+'_dis']]]),pd.concat([data_online_col_nona.loc[:,['id',col+'_dis']],data_online_col_na.loc[:,['id',col+'_dis']]])

def model_na_pred(data_online,col,predictors): #读取存好的缺失填充模型，然后预测
    data_online_col_na =  data_online.loc[pd.isnull( data_online.loc[:,col]),:]
    data_online_col_nona =  data_online.loc[pd.notnull( data_online.loc[:,col]),:]
    xgb1 = load_obj(col+'_xgb') #读取模型
    cutoffs = load_obj(col+'_cut') #保存切点
    data_online_col_nona[col+'_dis'] = np.digitize(data_online_col_nona[col], cutoffs, right=True)
    data_online_col_na_pred = xgb1.predict(data_online_col_na[predictors])
    data_online_col_na[col+'_dis'] = data_online_col_na_pred    
    return pd.concat([data_online_col_nona.loc[:,['id',col+'_dis']],data_online_col_na.loc[:,['id',col+'_dis']]])

def model_predict_valid(dtrain,dvalid,dtest,predictors,dep,donline):
    xgb1 = XGBClassifier(
     learning_rate =0.05,
     n_estimators=3000,
     max_depth=6,
     min_child_weight=1,
     gamma=0.1,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'binary:logistic',
     n_jobs=16,
     scale_pos_weight=1,
     seed=1,
     reg_alpha=0.5,
     reg_lambda =10,
     silent=False)
    xgb1.fit(dtrain[predictors],dtrain[dep],eval_set=[(dtrain[predictors],dtrain[dep]),(dvalid[predictors],dvalid[dep])],eval_metric='auc',early_stopping_rounds=50)
    featureimportance=pd.DataFrame(xgb1.feature_importances_)
    featureimportance['var'] = predictors
    df_fi = featureimportance.rename(columns={0:'fi'}).sort_values(by='fi',ascending=False)
    dtest_predprob = xgb1.predict_proba(dtest[predictors])[:,1]
    false_positive_rate, recall, thresholds = roc_curve(dtest[dep],dtest_predprob)
    roc_auc = auc(false_positive_rate, recall)
    at_score_valid = ant_score(dvalid[dep],xgb1.predict_proba(dvalid[predictors])[:,1])
    at_score = ant_score(dtest[dep],dtest_predprob)
    print ('at_score_valid：',at_score_valid)
    print ('at_score_test：',at_score)
    donline_predprob = xgb1.predict_proba(donline[predictors],ntree_limit=xgb1.best_iteration)[:,1]
    donline['score'] = donline_predprob
    return donline.loc[:,['id','score']],df_fi,dtest_predprob,[roc_auc,at_score,at_score_valid]


    #读取线下训练集和线上测试集
data_offline = pd.read_csv('data/train_data.csv')
data_online = pd.read_csv('data/test_b.csv')
#data_online = pd.read_csv('data/test_data.csv')

no_predictors = ['id','label','date']
predictors = [i for i in data_offline.columns if i not in no_predictors]

dep='label' 

date_sorted = sorted(list(set(data_offline['date'])))

#将无标签的样本当作bad
data_offline[dep] = data_offline[dep].apply(lambda x:1 if x==-1 else x)

# #手动剔除变量
# #a = [str(i) for i in range(20,25)]
# b = ['20','22','24','26','28','30','32','34','46','47','48','50','52','53']        
# c = [str(i) for i in range(64,72)]
# d = [str(i) for i in range(111,155)]

#drop_col = ['f'+i for i in (b+c+d)]
#predictors_3 = [i for i in predictors if i not in drop_col]


'''
不剔除变量，每10天作为一个模型的训练集，后12天作为测试和验证集(0.3856)
'''
test_data = data_offline.loc[data_offline.loc[:,'date'].isin(date_sorted[50:]),:]
data_online_proba_df_dic={}
df_fi_dic = {}
data_test_proba_dic = {}
score_dic = {}
for i in range(0,50,10):
    train_data = data_offline.loc[data_offline.loc[:,'date'].isin(date_sorted[i:i+10]),:]
    data_online_proba_df,df_fi,data_test_proba,score = model_predict(train_data,test_data,predictors,dep,data_online)
    data_online_proba_df_dic[i] = data_online_proba_df
    df_fi_dic[i] = df_fi
    data_test_proba_dic[i] = data_test_proba
    score_dic[i] = score
# {0: [0.958347999855191, 0.34400593471810087], 10: [0.9599038691637644, 0.31611275964391694], 
# 20: [0.9715198702915572, 0.36774480712166174], 30: [0.9721491601073059, 0.35691394658753706], 
# 40: [0.975960605025857, 0.38851632047477747]}


k = 0
j = 0    
for i in range(0,50,10):
    k = k+data_test_proba_dic[i]
    j = j+data_online_proba_df_dic[i]['score']
ant_score(test_data[dep],k) #394
false_positive_rate, recall, thresholds = roc_curve(test_data[dep],k)
auc(false_positive_rate, recall) #0.97

data_online['score'] = j
data_online['score'] = data_online['score'].apply(lambda x:decimal.Decimal(float(x/5)).quantize(decimal.Decimal('1.00000000000'), decimal.ROUND_DOWN))
data_online.loc[:,['id','score']].to_csv('data/data_online_proba_df_0701_2.csv',encoding='utf8',index=False)



'''
不剔除变量，前50天5折，后12天作为测试集(0.3517)
'''
test_data = data_offline.loc[data_offline.loc[:,'date'].isin(date_sorted[50:]),:]
date_tv = date_sorted[:50]
data_online_proba_df_dic={}
df_fi_dic = {}
data_test_proba_dic = {}
score_dic = {}
for i in range(0,50,10):
    date_valid = date_sorted[i:i+10]
    valid_data = data_offline.loc[data_offline.loc[:,'date'].isin(date_valid),:]
    date_train = [i for i in date_tv if i not in date_valid]
    train_data = data_offline.loc[data_offline.loc[:,'date'].isin(date_train),:]
    data_online_proba_df,df_fi,data_test_proba,score = model_predict_valid(train_data,valid_data,test_data,predictors,dep,data_online)
    data_online_proba_df_dic[i] = data_online_proba_df
    df_fi_dic[i] = df_fi
    data_test_proba_dic[i] = data_test_proba
    score_dic[i] = score
# {0: [0.9766713913698695, 0.41207715133531153], 10: [0.9776495544973037, 0.4114243323442136], 
# 20: [0.9778765404373234, 0.42261127596439174], 30: [0.9782054917618358, 0.41857566765578635], 
# 40: [0.972208288694092, 0.3999703264094955]}

k = 0
j = 0    
for i in range(0,50,10):
    k = k+data_test_proba_dic[i]
    j = j+data_online_proba_df_dic[i]['score']

ant_score(test_data[dep],k) #420
false_positive_rate, recall, thresholds = roc_curve(test_data[dep],k)
auc(false_positive_rate, recall) #0.977

data_online['score'] = j
data_online['score'] = data_online['score'].apply(lambda x:decimal.Decimal(float(x/5)).quantize(decimal.Decimal('1.00000000000'), decimal.ROUND_DOWN))
data_online.loc[:,['id','score']].to_csv('data/data_online_proba_df_0702.csv',encoding='utf8',index=False)


'''
剔除f20-f35,f46-53，每10天作为一个模型的训练集，后12天作为测试和验证集(base太低，舍弃)
'''
a = [str(i) for i in range(20,36)]
b = [str(i) for i in range(46,54)]
drop_col = ['f'+i for i in (a+b)]
predictors_3 = [i for i in predictors if i not in drop_col]

test_data = data_offline.loc[data_offline.loc[:,'date'].isin(date_sorted[50:]),:]
data_online_proba_df_dic={}
df_fi_dic = {}
data_test_proba_dic = {}
score_dic = {}
for i in range(0,50,10):
    train_data = data_offline.loc[data_offline.loc[:,'date'].isin(date_sorted[i:i+10]),:]
    data_online_proba_df,df_fi,data_test_proba,score = model_predict(train_data,test_data,predictors_3,dep,data_online)
    data_online_proba_df_dic[i] = data_online_proba_df
    df_fi_dic[i] = df_fi
    data_test_proba_dic[i] = data_test_proba
    score_dic[i] = score
# {0: [0.9385095294809208, 0.23827893175074183], 10: [0.9357700836701986, 0.2445994065281899], 
# 20: [0.9589291859029583, 0.29412462908011866], 30: [0.9568100322629732, 0.27201780415430266], 
# 40: [0.9624413331455726, 0.30136498516320476]}

k = 0
j = 0    
for i in range(0,50,10):
    k = k+data_test_proba_dic[i]
    j = j+data_online_proba_df_dic[i]['score']
ant_score(test_data[dep],k) #306
false_positive_rate, recall, thresholds = roc_curve(test_data[dep],k)
auc(false_positive_rate, recall) #0.961

data_online['score'] = j
data_online['score'] = data_online['score'].apply(lambda x:decimal.Decimal(float(x/5)).quantize(decimal.Decimal('1.00000000000'), decimal.ROUND_DOWN))
data_online.loc[:,['id','score']].to_csv('data/data_online_proba_df_0702_2.csv',encoding='utf8',index=False)

'''
以前40天作为训练集，40-50天作为valid,后12天作为测试集，不剔除变量(0.3423)
'''
test_data = data_offline.loc[data_offline.loc[:,'date'].isin(date_sorted[50:]),:]
valid_data = data_offline.loc[data_offline.loc[:,'date'].isin(date_sorted[40:50]),:]
train_data = data_offline.loc[data_offline.loc[:,'date'].isin(date_sorted[:40]),:]
data_online_proba_df,df_fi,data_test_proba,score = model_predict_valid(train_data,valid_data,test_data,predictors,dep,data_online)
df_fi.to_csv('data/df_fi_0702_2.csv',encoding='gbk')
#[0.972208288694092, 0.3999703264094955]

data_online['score'] = data_online_proba_df['score']
data_online['score'] = data_online['score'].apply(lambda x:decimal.Decimal(float(x/1)).quantize(decimal.Decimal('1.00000000000'), decimal.ROUND_DOWN))
data_online.loc[:,['id','score']].to_csv('data/data_online_proba_df_0702_2.csv',encoding='utf8',index=False)

'''
以前40天作为训练集，40-50天作为valid,后12天作为测试集，剔除f20-f35,f46-53(0.3183)
'''
a = [str(i) for i in range(20,36)]
b = [str(i) for i in range(46,54)]
drop_col = ['f'+i for i in (a+b)]
predictors_3 = [i for i in predictors if i not in drop_col]

test_data = data_offline.loc[data_offline.loc[:,'date'].isin(date_sorted[50:]),:]
valid_data = data_offline.loc[data_offline.loc[:,'date'].isin(date_sorted[40:50]),:]
train_data = data_offline.loc[data_offline.loc[:,'date'].isin(date_sorted[:40]),:]
data_online_proba_df,df_fi,data_test_proba,score = model_predict_valid(train_data,valid_data,test_data,predictors_3,dep,data_online)
df_fi.to_csv('data/df_fi_0703.csv',encoding='gbk')
#[0.9571722122103512, 0.2986053412462908]

data_online['score'] = data_online_proba_df['score']
data_online['score'] = data_online['score'].apply(lambda x:decimal.Decimal(float(x/1)).quantize(decimal.Decimal('1.00000000000'), decimal.ROUND_DOWN))
data_online.loc[:,['id','score']].to_csv('data/data_online_proba_df_0703.csv',encoding='utf8',index=False)


'''
剔除变量28-31，取前40天5折，后22天作为测试集(0.3824)
'''
a = [str(i) for i in range(28,36)]
#b = [str(i) for i in range(46,54)]
drop_col = ['f'+i for i in (a)]
predictors_3 = [i for i in predictors if i not in drop_col]

test_data = data_offline.loc[data_offline.loc[:,'date'].isin(date_sorted[40:]),:]
train_data = data_offline.loc[data_offline.loc[:,'date'].isin(date_sorted[:40]),:].reset_index(drop=True)

data_online_proba_df_dic={}
df_fi_dic = {}
data_test_proba_dic = {}
score_dic = {}

kf = KFold(n_splits=5,shuffle =True,random_state=999)
i=0
for trn,tst in kf.split(train_data.index):
    dtrain = train_data.loc[trn,:]
    dtest = train_data.loc[tst,:]
    data_online_proba_df,df_fi,data_test_proba,score = model_predict_valid(dtrain,dtest,test_data,predictors_3,dep,data_online)
    data_online_proba_df_dic[i] = data_online_proba_df
    df_fi_dic[i] = df_fi
    data_test_proba_dic[i] = data_test_proba
    score_dic[i] = score
    i+=1
# {0: [0.9739235564107266, 0.38860335195530726], 1: [0.9746367044473997, 0.3919712689545092], 
# 2: [0.9753575933066331, 0.3972067039106145], 3: [0.9740890024378781, 0.3890183559457302], 
# 4: [0.9760921458342038, 0.40209098164405427]}
    
k = 0
j = 0    
for i in range(0,5):
    k = k+data_test_proba_dic[i]
    j = j+data_online_proba_df_dic[i]['score']

ant_score(test_data[dep],k) #40145
false_positive_rate, recall, thresholds = roc_curve(test_data[dep],k)
auc(false_positive_rate, recall) #0.97

data_online['score'] = j
data_online['score'] = data_online['score'].apply(lambda x:decimal.Decimal(float(x/5)).quantize(decimal.Decimal('1.00000000000'), decimal.ROUND_DOWN))
data_online.loc[:,['id','score']].to_csv('data/data_online_proba_df_0703_2.csv',encoding='utf8',index=False)



'''
融合之前两种方法的结果(0.34+0.38->0.4056)
'''
data_online_proba_df_0701 = pd.read_csv('data/data_online_proba_df_0701.csv')
data_online_proba_df_0703_2 = pd.read_csv('data/data_online_proba_df_0703_2.csv')

#归一化
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_online_proba_df_0701['score_mm_01'] = scaler.fit_transform(np.array(data_online_proba_df_0701['score']).reshape(-1, 1))

scaler = MinMaxScaler()
data_online_proba_df_0703_2['score_mm_02'] = scaler.fit_transform(np.array(data_online_proba_df_0703_2['score']).reshape(-1, 1))

data_online_merge = data_online_proba_df_0701.merge(data_online_proba_df_0703_2,on='id')
data_online_merge['score'] = data_online_merge['score_mm_01']*0.4+data_online_merge['score_mm_02']*0.6
data_online_merge.loc[:,['id','score']].to_csv('data/data_online_proba_df_0704.csv',encoding='utf8',index=False)



'''
剔除变量28-31>1的样本，并剔除部分变量,取前40天5折，后22天作为测试集(0.4273)
'''
need_del = ['f20','f21','f22','f23','f24','f25','f26','f27',
'f32','f33','f34','f35',
'f46','f47','f48','f49','f50','f51','f52','f53',
'f64','f65','f66','f67','f68','f69','f70','f71',
'f111','f112','f113','f114','f115','f116','f117','f118','f119','f120','f121','f122','f123','f124','f125','f126','f127','f128','f129','f130','f131','f132','f133','f134','f135','f136','f137','f138','f139','f140','f141','f142','f143','f144','f145','f146','f147','f148','f149','f150','f151','f152','f153','f154','f161','f162','f163','f164','f165',
'f211','f212','f213','f214','f215','f216','f217','f218','f219','f220','f221','f222','f223','f224','f225','f226','f227','f228','f229','f230','f231','f232','f233']
predictors_3 = [i for i in predictors if i not in need_del]

data_offline_drop_id = set(data_offline.loc[((data_offline.loc[:,'f28']>1)|(data_offline.loc[:,'f29']>1)|(data_offline.loc[:,'f30']>1)|(data_offline.loc[:,'f31']>1)),:]['id'].tolist())
data_offline_filter = data_offline.loc[~data_offline.loc[:,'id'].isin(data_offline_drop_id),:]

test_data = data_offline_filter.loc[data_offline_filter.loc[:,'date'].isin(date_sorted[40:]),:]
train_data = data_offline_filter.loc[data_offline_filter.loc[:,'date'].isin(date_sorted[:40]),:].reset_index(drop=True)

data_online_proba_df_dic={}
df_fi_dic = {}
data_test_proba_dic = {}
score_dic = {}

kf = KFold(n_splits=5,shuffle =True,random_state=999)
i=0
for trn,tst in kf.split(train_data.index):
    dtrain = train_data.loc[trn,:]
    dtest = train_data.loc[tst,:]
    data_online_proba_df,df_fi,data_test_proba,score = model_predict_valid(dtrain,dtest,test_data,predictors_3,dep,data_online)
    data_online_proba_df_dic[i] = data_online_proba_df
    df_fi_dic[i] = df_fi
    data_test_proba_dic[i] = data_test_proba
    score_dic[i] = score
    i+=1


# {0: [0.9651598086389857, 0.3488900229650421, 0.46771447282252776], 
# 1: [0.9652473037834416, 0.3477672875733605, 0.4561035758323058], 
# 2: [0.9666281130653993, 0.35179892829803516, 0.44787031150667517], 
# 3: [0.9665505855290544, 0.3528195968359275, 0.4663716814159292], 
# 4: [0.96351887584895, 0.35412094922174026, 0.44806094182825484]}

    
k = 0
j = 0    
for i in range(0,5):
    k = k+data_test_proba_dic[i]
    j = j+data_online_proba_df_dic[i]['score']

ant_score(test_data[dep],k) #35
false_positive_rate, recall, thresholds = roc_curve(test_data[dep],k)
auc(false_positive_rate, recall) #96

data_online['score'] = j
data_online['score'] = data_online['score'].apply(lambda x:decimal.Decimal(float(x/5)).quantize(decimal.Decimal('1.00000000000'), decimal.ROUND_DOWN))
data_online.loc[:,['id','score']].to_csv('data/data_online_proba_df_0704_3.csv',encoding='utf8',index=False)


'''
融合之前两种方法的结果(0.34+0.4237->0.4357)
'''
data_online_proba_df_0701 = pd.read_csv('data/data_online_proba_df_0701.csv')
data_online_proba_df_0704_3 = pd.read_csv('data/data_online_proba_df_0704_3.csv')

#归一化
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_online_proba_df_0701['score_mm_01'] = scaler.fit_transform(np.array(data_online_proba_df_0701['score']).reshape(-1, 1))

scaler = MinMaxScaler()
data_online_proba_df_0704_3['score_mm_02'] = scaler.fit_transform(np.array(data_online_proba_df_0704_3['score']).reshape(-1, 1))

data_online_merge = data_online_proba_df_0701.merge(data_online_proba_df_0704_3,on='id')
data_online_merge['score'] = data_online_merge['score_mm_01']*0.3+data_online_merge['score_mm_02']*0.7
data_online_merge.loc[:,['id','score']].to_csv('data/data_online_proba_df_0706.csv',encoding='utf8',index=False)


'''
剔除变量28-31>1的样本，并剔除部分变量,按时间周期（月份）训练两个单模型,再融合0.34的分
'''
need_del = ['f20','f21','f22','f23','f24','f25','f26','f27',
'f32','f33','f34','f35',
'f46','f47','f48','f49','f50','f51','f52','f53',
'f64','f65','f66','f67','f68','f69','f70','f71',
'f111','f112','f113','f114','f115','f116','f117','f118','f119','f120','f121','f122','f123','f124','f125','f126','f127','f128','f129','f130','f131','f132','f133','f134','f135','f136','f137','f138','f139','f140','f141','f142','f143','f144','f145','f146','f147','f148','f149','f150','f151','f152','f153','f154','f161','f162','f163','f164','f165',
'f211','f212','f213','f214','f215','f216','f217','f218','f219','f220','f221','f222','f223','f224','f225','f226','f227','f228','f229','f230','f231','f232','f233']
predictors_3 = [i for i in predictors if i not in need_del]+['acc1']

data_offline_drop_id = set(data_offline.loc[((data_offline.loc[:,'f28']>1)|(data_offline.loc[:,'f29']>1)|(data_offline.loc[:,'f30']>1)|(data_offline.loc[:,'f31']>1)),:]['id'].tolist())
data_offline_filter = data_offline.loc[~data_offline.loc[:,'id'].isin(data_offline_drop_id),:]

test_data = data_offline_filter.loc[data_offline_filter.loc[:,'date'].isin(date_sorted[31:]),:].reset_index(drop=True)
train_data = data_offline_filter.loc[data_offline_filter.loc[:,'date'].isin(date_sorted[:31]),:].reset_index(drop=True)

#加一个排序特征
train_data = train_data.sort_values(by = 'date')
test_data = test_data.sort_values(by = 'date')
data_online = data_online.sort_values(by = 'date')

train_data['acc1'] = train_data.groupby(['f'+str(k) for k in range(6,20)]).cumcount()
test_data['acc1'] = test_data.groupby(['f'+str(k) for k in range(6,20)]).cumcount()
data_online['acc1'] = data_online.groupby(['f'+str(k) for k in range(6,20)]).cumcount()

#前31天
data_online_proba_df_dic={}
df_fi_dic = {}
data_test_proba_dic = {}
score_dic = {}

kf = KFold(n_splits=5,shuffle =True,random_state=999)
i=0
for trn,tst in kf.split(train_data.index):
    dtrain = train_data.loc[trn,:]
    dtest = train_data.loc[tst,:]
    data_online_proba_df,df_fi,data_test_proba,score = model_predict_valid(dtrain,dtest,test_data,predictors_3,dep,data_online)
    data_online_proba_df_dic[i] = data_online_proba_df
    df_fi_dic[i] = df_fi
    data_test_proba_dic[i] = data_test_proba
    score_dic[i] = score
    i+=1
# {0: [0.9645758707612707, 0.3331730769230769, 0.4631372549019608], 
# 1: [0.9647035603452045, 0.3281989644970414, 0.4802409638554217], 
# 2: [0.9654974008369382, 0.32572115384615385, 0.4751612903225806], 
# 3: [0.9651684428485305, 0.3317492603550296, 0.4489949748743719], 
# 4: [0.9650863129778878, 0.3267751479289941, 0.4530201342281879]}
    
k = 0
j = 0    
for i in range(0,5):
    k = k+data_test_proba_dic[i]
    j = j+data_online_proba_df_dic[i]['score']

ant_score(test_data[dep],k) #33
false_positive_rate, recall, thresholds = roc_curve(test_data[dep],k)
auc(false_positive_rate, recall) #966

data_online['score'] = j
data_online['score'] = data_online['score'].apply(lambda x:decimal.Decimal(float(x/5)).quantize(decimal.Decimal('1.00000000000'), decimal.ROUND_DOWN))
data_online.loc[:,['id','score']].to_csv('data/data_online_proba_df_0708_x.csv',encoding='utf8',index=False)

#后31天
data_online_proba_df_dic_={}
df_fi_dic_ = {}
data_test_proba_dic_ = {}
score_dic_ = {}

kf_ = KFold(n_splits=5,shuffle =True,random_state=999)
i=0
for trn,tst in kf_.split(test_data.index):
    dtrain = test_data.loc[trn,:]
    dtest = test_data.loc[tst,:]
    data_online_proba_df_,df_fi_,data_test_proba_,score_ = model_predict_valid(dtrain,dtest,train_data,predictors_3,dep,data_online)
    data_online_proba_df_dic_[i] = data_online_proba_df_
    df_fi_dic_[i] = df_fi_
    data_test_proba_dic_[i] = data_test_proba_
    score_dic_[i] = score_
    i+=1

# {0: [0.9654420699288287, 0.3174910510901399, 0.44389587073608616], 
# 1: [0.9653507995952759, 0.32357630979498864, 0.4105009633911368], 
# 2: [0.9638684677728004, 0.30704523267165634, 0.41570915619389587], 
# 3: [0.9642621489494869, 0.30890009762447124, 0.43913043478260866], 
# 4: [0.9652630414328518, 0.3095509274324764, 0.4371347785108388]}


k_ = 0
j_ = 0    
for i in range(0,5):
    k_ = k_+data_test_proba_dic_[i]
    j_ = j_+data_online_proba_df_dic_[i]['score']

ant_score(train_data[dep],k_) #32
false_positive_rate, recall, thresholds = roc_curve(train_data[dep],k_)
auc(false_positive_rate, recall) #966

data_online['score'] = j_
data_online['score'] = data_online['score'].apply(lambda x:decimal.Decimal(float(x/5)).quantize(decimal.Decimal('1.00000000000'), decimal.ROUND_DOWN))
data_online.loc[:,['id','score']].to_csv('data/data_online_proba_df_0708_y.csv',encoding='utf8',index=False)


#归一化
data_online_proba_df_0708_x = pd.read_csv('data/data_online_proba_df_0708_x.csv')
data_online_proba_df_0708_y = pd.read_csv('data/data_online_proba_df_0708_y.csv')

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_online_proba_df_0708_x['score_mm_01'] = scaler.fit_transform(np.array(data_online_proba_df_0708_x['score']).reshape(-1, 1))

scaler = MinMaxScaler()
data_online_proba_df_0708_y['score_mm_02'] = scaler.fit_transform(np.array(data_online_proba_df_0708_y['score']).reshape(-1, 1))

data_online_merge = data_online_proba_df_0708_x.merge(data_online_proba_df_0708_y,on='id')
data_online_merge['score'] = data_online_merge['score_mm_01']*0.7+data_online_merge['score_mm_02']*0.3
data_online_merge.loc[:,['id','score']].to_csv('data/data_online_proba_df_0708_z.csv',encoding='utf8',index=False)


#再次融合归一化,融合0.34的分
data_online_proba_df_0701 = pd.read_csv('data/data_online_proba_df_0701.csv')
data_online_proba_df_0708_z = pd.read_csv('data/data_online_proba_df_0708_z.csv')

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_online_proba_df_0701['score_mm_01'] = scaler.fit_transform(np.array(data_online_proba_df_0701['score']).reshape(-1, 1))

scaler = MinMaxScaler()
data_online_proba_df_0708_z['score_mm_02'] = scaler.fit_transform(np.array(data_online_proba_df_0708_z['score']).reshape(-1, 1))

data_online_merge = data_online_proba_df_0701.merge(data_online_proba_df_0708_z,on='id')
data_online_merge['score'] = data_online_merge['score_mm_01']*0.3+data_online_merge['score_mm_02']*0.7
data_online_merge.loc[:,['id','score']].to_csv('data/data_online_proba_df_0708.csv',encoding='utf8',index=False)




