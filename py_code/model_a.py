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
     n_estimators=2000,
     max_depth=5,
     min_child_weight=1,
     gamma=0.1,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'binary:logistic',
     n_jobs=16,
     scale_pos_weight=1,
     seed=1,
     reg_alpha=0.1,
     silent=False)
    xgb1.fit(dtrain[predictors],dtrain[dep],eval_set=[(dtrain[predictors],dtrain[dep]),(dtest[predictors],dtest[dep])],eval_metric='auc',early_stopping_rounds=50)
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
         learning_rate =0.1,
         n_estimators=2000,
         max_depth=5,
         min_child_weight=1,
         gamma=0.1,
         subsample=0.8,
         colsample_bytree=0.8,
         objective= 'binary:logistic',
         n_jobs=16,
         scale_pos_weight=1,
         seed=1,
         reg_alpha=0.1,
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
     n_estimators=2000,
     max_depth=5,
     min_child_weight=1,
     gamma=0.1,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'multi:softmax',
     n_jobs=16,
     scale_pos_weight=1,
     seed=1,
     reg_alpha=0.1,
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
    cutoffs = load_obj(col+'_cut') #duqu切点
    data_online_col_nona[col+'_dis'] = np.digitize(data_online_col_nona[col], cutoffs, right=True)
    data_online_col_na_pred = xgb1.predict(data_online_col_na[predictors])
    data_online_col_na[col+'_dis'] = data_online_col_na_pred    
    return pd.concat([data_online_col_nona.loc[:,['id',col+'_dis']],data_online_col_na.loc[:,['id',col+'_dis']]])


    #读取线下训练集和线上测试集
data_offline = pd.read_csv('data/train_data.csv')
data_online = pd.read_csv('data/test_b.csv')
#data_online = pd.read_csv('data/test_data.csv')

no_predictors = ['id','label','date']
predictors = [i for i in data_offline.columns if i not in no_predictors]

dep='label' 

date_sorted = sorted(list(set(data_offline['date'])))
data_offline_filter = data_offline.loc[data_offline.loc[:,'label'].isin([0,1]),:]

#手动剔除变量
#a = [str(i) for i in range(20,25)]
b = ['20','22','24','26','28','30','32','34','46','47','48','50','52','53']        
c = [str(i) for i in range(64,72)]
d = [str(i) for i in range(111,155)]

drop_col = ['f'+i for i in (b+c+d)]
predictors_3 = [i for i in predictors if i not in drop_col]
    
    
'''
按天给所有的特征做标准化,效果很差
'''


#按天给样本做归一化
from sklearn.preprocessing import StandardScaler

offline_date = set(data_offline.date.tolist())

data_offline_trans = pd.DataFrame()
for dt in offline_date:
    dt_offline = data_offline.loc[data_offline.loc[:,'date']==dt,:]
    dt_offline.fillna(dt_offline.mean().to_dict(),inplace=True)
    scaler = StandardScaler()
    dt_offline_trans = pd.DataFrame(scaler.fit_transform(dt_offline[predictors]),columns=predictors)
    dt_offline_trans['label'] = dt_offline['label'].values
    data_offline_trans = data_offline_trans.append(dt_offline_trans)
    
online_date = set(data_online.date.tolist())
data_online_trans = pd.DataFrame()
for dt in online_date:
    dt_online = data_online.loc[data_online.loc[:,'date']==dt,:]
    dt_online.fillna(dt_online.mean().to_dict(),inplace=True)
    scaler = StandardScaler()
    dt_online_trans = pd.DataFrame(scaler.fit_transform(dt_online[predictors]),columns=predictors)
    dt_online_trans['id'] = dt_online['id'].values
    data_online_trans = data_online_trans.append(dt_online_trans)

 

data_offline_trans['label'] = data_offline_trans['label'].apply(lambda x:0 if x==0 else 1)
 
data_offline_trans_train = data_offline_trans.head(800000)

data_offline_trans_test = data_offline_trans.tail(194731)


#随机森林
from sklearn.ensemble import RandomForestClassifier
rf2 = RandomForestClassifier(n_estimators =1000,n_jobs =16,random_state =3,max_depth =5)
rf2.fit(data_offline_trans_train[predictors],data_offline_trans_train[dep])
test_proba = rf2.predict_proba(data_offline_trans_test[predictors])[:,1]
false_positive_rate, recall, thresholds = roc_curve(data_offline_trans_test[dep],test_proba)
print(auc(false_positive_rate, recall))
print (ant_score(data_offline_trans_test[dep],test_proba)) #0.275

data_online_proba = rf2.predict_proba(data_online_trans[predictors])[:,1]

data_online_proba_df = pd.DataFrame(pd.Series(data_online_proba)).rename(columns={0:'score'})
data_online_proba_df['id'] = data_online['id'].tolist()
data_online_proba_df.loc[:,['id','score']].to_csv('data/data_online_proba_df(scaler_model_0521).csv',encoding='utf8',index=False) #0.000xx

'''
剔除部分特征，剔除部分波动较大的特征
'''

#剔除部分特征
data_offline_good_std_mean_ratio = pd.read_csv('data/data_offline_good_std_mean_ratio.csv',names=['var','std_mean']).sort_values(by='std_mean',ascending=False)
data_offline_good_std_mean_ratio_top80 = data_offline_good_std_mean_ratio.head(80)['var'].tolist()

predictors_2 = [i for i in predictors if i not in data_offline_good_std_mean_ratio_top80]

date_sorted = sorted(list(set(data_offline['date'])))
data_offline_filter = data_offline.loc[data_offline.loc[:,'label'].isin([0,1]),:]
dep='label'

    
ss = model_cv(data_offline_filter,dep=dep,predictors=predictors)
#{0: [0.9792194666794376, 0.467163252638113], 
#10: [0.9881994452258299, 0.5711595055915244], 
#20: [0.9882002981616977, 0.4643852978453739], 
#30: [0.9876713122027531, 0.5674801362088535], 
#40: [0.9898105330295741, 0.5684412470023981], 
#50: [0.982605723976829, 0.5163246268656716], 
#60: [0.9898985610820151, 0.6077092511013216]}

ss_filter = model_cv(data_offline_filter,dep=dep,predictors=predictors_2)
#{0: [0.9785181771508855, 0.4247672253258845], 
#10: [0.9862339322254076, 0.5310182460270747], 
#20: [0.9861110099626609, 0.4247993240388678], 
#30: [0.986270294937236, 0.5171396140749149], 
#40: [0.9885248377674911, 0.5381774580335731], 
#50: [0.9786234612644469, 0.45811567164179107]}

#手动剔除变量
#a = [str(i) for i in range(20,25)]
b = ['20','22','24','26','32','34','48','50','52','53']        
c = [str(i) for i in range(64,72)]
d = [str(i) for i in range(111,155)]

drop_col = ['f'+i for i in (b+c+d)]
predictors_3 = [i for i in predictors if i not in drop_col]     
ss_3 =  model_cv(data_offline_filter,dep=dep,predictors=predictors_3)  
#{0: [0.9774609498890061, 0.4610800744878957], 
#10: [0.985878061891291, 0.5371983519717481], 
#20: [0.9882799376077609, 0.449133924799324], 
#30: [0.98699509603163, 0.5442111237230419], 
#40: [0.9897590242993919, 0.559568345323741], 
#50: [0.9825429923253275, 0.49827425373134326]}


'''
将最早的前20天作为测试集,取手工剔除的变量，取predictors_3
'''  


data_online_proba_df_0523,df_fi_0523 = model_predict(train_data,test_data,predictors_3,dep,data_online) #0.4766
df_fi_0523 = df_fi.rename(columns={0:'fi'}).sort_values(by='fi',ascending=False)
df_fi_0523.to_csv('data/df_fi_0523.csv',encoding='utf8',index=False)

data_online_proba_df_0523['score'] = data_online_proba_df_0523['score'].apply(lambda x:decimal.Decimal(float(x)).quantize(decimal.Decimal('1.000000000'), decimal.ROUND_DOWN))
data_online_proba_df_0523.loc[:,['id','score']].to_csv('data/data_online_proba_df_0523.csv',encoding='utf8',index=False)

'''
以predictors_3为准，去掉线性相关高的变量，去掉与测试集量级较大或分布不均衡的变量
'''
e = [str(i) for i in range(9,20)]+['21','23','35','33','49','51','296']+[str(i) for i in range(98,102)]+[str(i) for i in range(278,280)]+[str(i) for i in range(281,286)]
drop_col_2 = ['f'+i for i in (e+b+c+d)]
predictors_4 = [i for i in predictors if i not in drop_col_2] 

df_corr = data_offline.fillna(0).loc[:,predictors_4].corr().applymap(lambda x: abs(x))
corr_high_var = corr_filter(df_corr,0.97)
predictors_5 = [i for i in predictors_4 if i not in corr_high_var] 

ss_4 = model_cv(data_offline_filter,dep=dep,predictors=predictors_5)

# {0: [0.9744821804652003, 0.42960893854748605], 
# 10: [0.9866095742974568, 0.5151265450264861], 
# 20: [0.9860257232988919, 0.38069286016054077], 
# 30: [0.9856513984171109, 0.4917707150964813], 
# 40: [0.9886581493909524, 0.5056594724220624], 
# 50: [0.9820792213684602, 0.4466417910447761]}

data_online_proba_df_0524,df_fi_0524 = model_predict(train_data,test_data,predictors_5,dep,data_online) #0.45
df_fi_0524.to_csv('data/df_fi_0524.csv',encoding='utf8',index=False)
data_online_proba_df_0524['score'] = data_online_proba_df_0524['score'].apply(lambda x:decimal.Decimal(float(x)).quantize(decimal.Decimal('1.000000000'), decimal.ROUND_DOWN))
data_online_proba_df_0524.loc[:,['id','score']].to_csv('data/data_online_proba_df_0524.csv',encoding='utf8',index=False)


'''
将最后六天作为test，所以现在有56天的数据，分成8*7天,剔除部分特征(0.3095)
'''
test_data = data_offline_filter.loc[data_offline_filter.loc[:,'date'].isin(date_sorted[56:]),:]
data_online_proba_df_dic={}
df_fi_dic = {}
data_test_proba_dic = {}
score_dic = {}
for i in range(0,56,7):
    train_data = data_offline_filter.loc[data_offline_filter.loc[:,'date'].isin(date_sorted[i:i+7]),:]
    data_online_proba_df,df_fi,data_test_proba,score = model_predict(train_data,test_data,predictors_3,dep,data_online)
    data_online_proba_df_dic[i] = data_online_proba_df
    df_fi_dic[i] = df_fi
    data_test_proba_dic[i] = data_test_proba
    score_dic[i] = score

k = 0
j = 0    
for i in range(0,56,7):
    k = k+data_test_proba_dic[i]
    j = j+data_online_proba_df_dic[i]['score']
ant_score(test_data[dep],k) #0.49
false_positive_rate, recall, thresholds = roc_curve(test_data[dep],k)
auc(false_positive_rate, recall)

data_online['score'] = j
data_online['score'] = data_online['score'].apply(lambda x:decimal.Decimal(float(x/8)).quantize(decimal.Decimal('1.00000000000'), decimal.ROUND_DOWN))
data_online.loc[:,['id','score']].to_csv('data/data_online_proba_df_0529.csv',encoding='utf8',index=False)

'''
将最后六天作为test，所以现在有56天的数据，分成8*7天,不剔除特征(0.2983)
'''
test_data = data_offline_filter.loc[data_offline_filter.loc[:,'date'].isin(date_sorted[56:]),:]
data_online_proba_df_dic={}
df_fi_dic = {}
data_test_proba_dic = {}
score_dic = {}
for i in range(0,56,7):
    train_data = data_offline_filter.loc[data_offline_filter.loc[:,'date'].isin(date_sorted[i:i+7]),:]
    data_online_proba_df,df_fi,data_test_proba,score = model_predict(train_data,test_data,predictors,dep,data_online)
    data_online_proba_df_dic[i] = data_online_proba_df
    df_fi_dic[i] = df_fi
    data_test_proba_dic[i] = data_test_proba
    score_dic[i] = score

k = 0
j = 0    
for i in range(0,56,7):
    k = k+data_test_proba_dic[i]
    j = j+data_online_proba_df_dic[i]['score']
ant_score(test_data[dep],k) #0.49
false_positive_rate, recall, thresholds = roc_curve(test_data[dep],k)
auc(false_positive_rate, recall)

data_online['score'] = j
data_online['score'] = data_online['score'].apply(lambda x:decimal.Decimal(float(x/8)).quantize(decimal.Decimal('1.00000000000'), decimal.ROUND_DOWN))
data_online.loc[:,['id','score']].to_csv('data/data_online_proba_df_0530.csv',encoding='utf8',index=False)


'''
将最后六天作为test，所以现在有56天的数据，分成8*7天,用test中的一天作为valid,剔除部分特征，对某些变量做差分，（0.363）
'''
norm_col = [21,23,25,27,33,35,49,51,57,58,61,62,63,72,73,74,75,80,81,90,91,98,99,100,101,104,105,106,109,110,161,165,177,208,209,210]+c+d
coll = ['f'+str(i) for i in norm_col]
   
#merge
data_offline_filter_avg_diff = avg_diff(data_offline_filter,coll)
data_offline_filter_med_diff = med_diff(data_offline_filter,coll)
data_offline_filter_merge = data_offline_filter.merge(data_offline_filter_avg_diff).merge(data_offline_filter_med_diff)

data_online_avg_diff = avg_diff(data_online,coll)
data_online_med_diff = med_diff(data_online,coll)
data_online_merge = data_online.merge(data_online_avg_diff).merge(data_online_med_diff)

#将最后六天作为test，所以现在有56天的数据，分成8*7天,剔除部分特征，并加一部分差分特征
predictors_6 = predictors_3+[i+'_median_diff' for i in coll]+[i+'_avg_diff' for i in coll]
test_data = data_offline_filter_merge.loc[data_offline_filter_merge.loc[:,'date'].isin(date_sorted[56:]),:]
data_online_proba_df_dic={}
df_fi_dic = {}
data_test_proba_dic = {}
score_dic = {}

for i in range(0,56,7):
    train_data = data_offline_filter_merge.loc[data_offline_filter_merge.loc[:,'date'].isin(date_sorted[i:i+7]),:]
    data_online_proba_df,df_fi,data_test_proba,score = model_predict(train_data,test_data,predictors_6,dep,data_online_merge)
    data_online_proba_df_dic[i] = data_online_proba_df
    df_fi_dic[i] = df_fi
    data_test_proba_dic[i] = data_test_proba
    score_dic[i] = score
#score {0: [0.967772391802948, 0.4050147492625369], 7: [0.9639375579586887, 0.3911504424778761], 14: [0.9744207080892142, 0.4018436578171091], 
# 21: [0.9777830043249836, 0.4202064896755162], 28: [0.9784440671056455, 0.42551622418879054], 35: [0.971776148445519, 0.396386430678466], 
# 42: [0.9791686664097965, 0.46084070796460175], 49: [0.9814549917305824, 0.4615044247787611]}


k = 0
j = 0    
for i in range(0,56,7):
    k = k+data_test_proba_dic[i]
    j = j+data_online_proba_df_dic[i]['score']
ant_score(test_data[dep],k) #0.48
false_positive_rate, recall, thresholds = roc_curve(test_data[dep],k)
auc(false_positive_rate, recall)

data_online['score'] = j
data_online['score'] = data_online['score'].apply(lambda x:decimal.Decimal(float(x/8)).quantize(decimal.Decimal('1.00000000000'), decimal.ROUND_DOWN))
data_online.loc[:,['id','score']].to_csv('data/data_online_proba_df_0531.csv',encoding='utf8',index=False)




'''
将最后六天作为test，所以现在有56天的数据，分成4*14天,剔除部分特征，对某些变量做差分，(用全部的bad标签不可行，0.3527)
'''
#改写上部分的训练部分

#将最后六天作为test，所以现在有56天的数据，分成4*14天,但每次bad标签都用于训练，剔除部分特征，并加一部分差分特征
predictors_6 = predictors_3+[i+'_median_diff' for i in coll]+[i+'_avg_diff' for i in coll]
test_data = data_offline_filter_merge.loc[data_offline_filter_merge.loc[:,'date'].isin(date_sorted[56:]),:]
#train_data_all = data_offline_filter_merge.loc[data_offline_filter_merge.loc[:,'date'].isin(date_sorted[:56]),:]

#good_train_data_all = train_data_all.loc[train_data_all.loc[:,'label'].isin([0])]
#bad_train_data_all = train_data_all.loc[train_data_all.loc[:,'label'].isin([1])]

data_online_proba_df_dic={}
df_fi_dic = {}
data_test_proba_dic = {}
score_dic = {}

for i in range(0,56,4):
    train_data = data_offline_filter_merge.loc[data_offline_filter_merge.loc[:,'date'].isin(date_sorted[i:i+4]),:]
    #train_data = pd.concat([good_train_data,bad_train_data_all])
    data_online_proba_df,df_fi,data_test_proba,score = model_predict(train_data,test_data,predictors_6,dep,data_online_merge)
    data_online_proba_df_dic[i] = data_online_proba_df
    df_fi_dic[i] = df_fi
    data_test_proba_dic[i] = data_test_proba
    score_dic[i] = score
# {0: [0.965312135860509, 0.38325958702064894], 4: [0.9461026184045838, 0.3573008849557522], 8: [0.9509311646801976, 0.3294247787610619], 
# 12: [0.9572091766981909, 0.35656342182890854], 16: [0.9534950084455003, 0.35250737463126847], 20: [0.9744840978770832, 0.3615781710914454], 
# 24: [0.9748909205024939, 0.4011061946902655], 28: [0.9768658803624005, 0.4129793510324484], 32: [0.9771843041828502, 0.3769911504424779], 
# 36: [0.9704987998230262, 0.3533185840707964], 40: [0.9769203380897045, 0.42175516224188786], 44: [0.9765783312213939, 0.4036873156342182], 
# 48: [0.9787548719612069, 0.3992625368731564], 52: [0.9802967773817771, 0.4539823008849557]}

k = 0
j = 0    
for i in range(0,56,4):
    k = k+data_test_proba_dic[i]
    j = j+data_online_proba_df_dic[i]['score']
ant_score(test_data[dep],k) #0.479
false_positive_rate, recall, thresholds = roc_curve(test_data[dep],k)
auc(false_positive_rate, recall)

data_online['score'] = j
data_online['score'] = data_online['score'].apply(lambda x:decimal.Decimal(float(x/14)).quantize(decimal.Decimal('1.00000000000'), decimal.ROUND_DOWN))
data_online.loc[:,['id','score']].to_csv('data/data_online_proba_df_0601.csv',encoding='utf8',index=False)

'''
填充缺失值，64-154
'''
#将offline当做训练集，online当做测试集，offline中的最后6天当做valid
# na_col = ['f'+str(i) for i in  range(64,155)]
# col = 'f64'

# data_offline_filter_col_nona = data_offline_filter.loc[pd.notnull(data_offline_filter.loc[:,col]),:]
# data_online_col_nona = data_online.loc[pd.notnull(data_online.loc[:,col]),:]

# train_col = data_offline_filter_col_nona.loc[data_offline_filter_col_nona.loc[:,'date'].isin(date_sorted[:40]),:]
# valid_col = data_offline_filter_col_nona.loc[data_offline_filter_col_nona.loc[:,'date'].isin(date_sorted[40:46]),:]
# test_col = data_online_col_nona

# xgb1 = XGBRegressor(
 # learning_rate =0.1,
 # n_estimators=2000,
 # max_depth=5,
 # min_child_weight=1,
 # gamma=0.1,
 # subsample=0.8,
 # colsample_bytree=0.8,
 # n_jobs=16,
 # scale_pos_weight=1,
 # seed=1,
 # reg_alpha=0.1,
 # silent=False)

# predictors = [i for i in predictors if i not in  na_col] 
# dep = col
# dtrain = train_col
# dtest = valid_col
 
# xgb1.fit(dtrain[predictors],dtrain[dep],eval_set=[(dtrain[predictors],dtrain[dep]),(dtest[predictors],dtest[dep])],eval_metric='mae',early_stopping_rounds=10)



# featureimportance=pd.DataFrame(xgb1.feature_importances_)
# featureimportance['var'] = predictors
# df_fi = featureimportance.rename(columns={0:'fi'}).sort_values(by='fi',ascending=False)
# dtest_pred = xgb1.predict(test_col[predictors])
# mae_loss = mean_absolute_error(test_col[dep],dtest_pred)

#col_qcut = pd.qcut(data_offline_filter[col],10,duplicates ='drop')


#由于回归精度不佳，改为分类，先离散，默认十组，等频
na_col = ['f'+str(i) for i in  range(64,155)]
# col = 'f64'

# data_offline_filter_col_nona = data_offline_filter.loc[pd.notnull(data_offline_filter.loc[:,col]),:]
# data_online_col_nona = data_online.loc[pd.notnull(data_online.loc[:,col]),:]

# k= pd.qcut(data_offline_filter_col_nona[col].tolist()+[0], 10, retbins=True, labels=False,duplicates ='drop')
# cutoffs = k[1]

# data_offline_filter_col_nona[col] = np.digitize(data_offline_filter_col_nona[col], cutoffs, right=True)
# data_online_col_nona[col] = np.digitize(data_online_col_nona[col], cutoffs, right=True)

# train_col = data_offline_filter_col_nona.loc[data_offline_filter_col_nona.loc[:,'date'].isin(date_sorted[:40]),:]
# valid_col = data_offline_filter_col_nona.loc[data_offline_filter_col_nona.loc[:,'date'].isin(date_sorted[40:46]),:]
# test_col = data_online_col_nona


# xgb1 = XGBClassifier(
 # learning_rate =0.05,
 # n_estimators=2000,
 # max_depth=5,
 # min_child_weight=1,
 # gamma=0.1,
 # subsample=0.8,
 # colsample_bytree=0.8,
 # objective= 'multi:softmax',
 # n_jobs=16,
 # scale_pos_weight=1,
 # seed=1,
 # reg_alpha=0.1,
 # silent=False)
 
# predictors = [i for i in predictors if i not in  na_col] 
# dep = col
# dtrain = train_col
# dtest = valid_col
 
# xgb1.fit(dtrain[predictors],dtrain[dep],eval_set=[(dtrain[predictors],dtrain[dep]),(dtest[predictors],dtest[dep])],eval_metric='mlogloss',early_stopping_rounds=10)

# dtest_pred = xgb1.predict(test_col[predictors])

def model_na_pred(data_offline_filter,data_online,col,predictors):
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
     n_estimators=2000,aw
     max_depth=5,
     min_child_weight=1,
     gamma=0.1,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'multi:softmax',
     n_jobs=16,
     scale_pos_weight=1,
     seed=1,
     reg_alpha=0.1,
     silent=False)
    xgb1.fit(train_col[predictors],train_col[dep],eval_set=[(train_col[predictors],train_col[dep]),(valid_col[predictors],valid_col[dep])],eval_metric='mlogloss',early_stopping_rounds=10)
    data_offline_filter_col_na_pred = xgb1.predict(data_offline_filter_col_na[predictors])
    data_online_col_na_pred = xgb1.predict(data_online_col_na[predictors])
    data_offline_filter_col_na[col+'_dis'] = data_offline_filter_col_na_pred
    data_online_col_na[col+'_dis'] = data_online_col_na_pred
    
    return pd.concat([data_offline_filter_col_nona.loc[:,['id',col+'_dis']],data_offline_filter_col_na.loc[:,['id',col+'_dis']]]),pd.concat([data_online_col_nona.loc[:,['id',col+'_dis']],data_online_col_na.loc[:,['id',col+'_dis']]])

for coc in na_col: #速度极慢，跑了24小时没跑完跑到包括f106，后面先筛选
    col_offline,col_online = model_na_pred(data_offline_filter,data_online,coc,predictors)
    data_offline_filter = pd.merge(data_offline_filter,col_offline,on='id')
    data_online = pd.merge(data_online,col_online,on='id')

#data_offline_filter.to_csv('data/data_offline_filter_na_pred.csv',index=False)
#data_online.to_csv('data/data_online_na_pred.csv',index=False)
    
    
norm_col = [21,23,25,27,33,35,49,51,57,58,61,62,63,72,73,74,75,80,81,90,91,98,99,100,101,104,105,106,109,110,161,165,177,208,209,210]+c+d
coll = ['f'+str(i) for i in norm_col]
   
#merge
data_offline_filter_avg_diff = avg_diff(data_offline_filter,coll)
data_offline_filter_med_diff = med_diff(data_offline_filter,coll)
data_offline_filter_merge = data_offline_filter.merge(data_offline_filter_avg_diff).merge(data_offline_filter_med_diff)

data_online_avg_diff = avg_diff(data_online,coll)
data_online_med_diff = med_diff(data_online,coll)
data_online_merge = data_online.merge(data_online_avg_diff).merge(data_online_med_diff)

#将最后六天作为test，所以现在有56天的数据，分成8*7天,剔除部分特征，并加一部分差分特征和离散预测特征（0.3936）
predictors_7 = predictors_3+[i+'_median_diff' for i in coll]+[i+'_avg_diff' for i in coll]+['f'+str(i)+'_dis' for i in  range(64,107)]
test_data = data_offline_filter_merge.loc[data_offline_filter_merge.loc[:,'date'].isin(date_sorted[56:]),:]
data_online_proba_df_dic={}
df_fi_dic = {}
data_test_proba_dic = {}
score_dic = {}
dep = 'label'

for i in range(0,56,7):
    train_data = data_offline_filter_merge.loc[data_offline_filter_merge.loc[:,'date'].isin(date_sorted[i:i+7]),:]
    data_online_proba_df,df_fi,data_test_proba,score = model_predict(train_data,test_data,predictors_7,dep,data_online_merge)
    data_online_proba_df_dic[i] = data_online_proba_df
    df_fi_dic[i] = df_fi
    data_test_proba_dic[i] = data_test_proba
    score_dic[i] = score
# {0: [0.9699650132617397, 0.40530973451327434], 7: [0.9663847017566164, 0.38930678466076696], 14: [0.9753417111395994, 0.39085545722713866], 
# 21: [0.9777082180706225, 0.41718289085545723], 28: [0.9791575220275751, 0.41615044247787614], 35: [0.9777346643443008, 0.4066371681415929], 
# 42: [0.9795477296658518, 0.4526548672566372], 49: [0.9820750099245954, 0.46438053097345133]}


k = 0
j = 0    
for i in range(0,56,7):
    k = k+data_test_proba_dic[i]
    j = j+data_online_proba_df_dic[i]['score']
    
ant_score(test_data[dep],k) #0.478
false_positive_rate, recall, thresholds = roc_curve(test_data[dep],k)
auc(false_positive_rate, recall)#0.981

data_online['score'] = j
data_online['score'] = data_online['score'].apply(lambda x:decimal.Decimal(float(x/8)).quantize(decimal.Decimal('1.00000000000'), decimal.ROUND_DOWN))
data_online.loc[:,['id','score']].to_csv('data/data_online_proba_df_0603.csv',encoding='utf8',index=False)


'''
沿用上一部分，不过先去除相关性过高的变量，填充缺失值，64-154
'''
na_col = ['f'+str(i) for i in  range(64,155)]
train_col = data_offline.loc[data_offline.loc[:,'date'].isin(date_sorted[:46]),:]
df_corr_na_col = train_col.fillna(0).loc[:,na_col].corr().applymap(lambda x: abs(x))
corr_high_var = corr_filter(df_corr_na_col,0.97)
na_col_filter = [i for i in na_col if i not in corr_high_var]  #只剩下39个变量
save_obj(na_col_filter,'na_col_filter')

predictors_na = [i for i in predictors if i not in  na_col] 

import copy 
data_offline_filter_cp = copy.deepcopy(data_offline_filter)
data_online_cp = copy.deepcopy(data_online)

for coc in na_col_filter: 
    col_offline,col_online = model_na_train(data_offline_filter_cp,data_online_cp,coc,predictors_na)
    data_offline_filter = pd.merge(data_offline_filter,col_offline,on='id')
    data_online = pd.merge(data_online,col_online,on='id')

#data_offline_filter.to_csv('data/data_offline_filter_na_pred_filter.csv',index=False)
#data_online.to_csv('data/data_online_na_pred_filter.csv',index=False)
    
    
norm_col = [21,23,25,27,33,35,49,51,57,58,61,62,63,72,73,74,75,80,81,90,91,98,99,100,101,104,105,106,109,110,161,165,177,208,209,210]+c+d
coll = ['f'+str(i) for i in norm_col]
   
#merge
data_offline_filter_avg_diff = avg_diff(data_offline_filter,coll)
data_offline_filter_med_diff = med_diff(data_offline_filter,coll)
data_offline_filter_merge = data_offline_filter.merge(data_offline_filter_avg_diff).merge(data_offline_filter_med_diff)

data_online_avg_diff = avg_diff(data_online,coll)
data_online_med_diff = med_diff(data_online,coll)
data_online_merge = data_online.merge(data_online_avg_diff).merge(data_online_med_diff)

#将最后六天作为test，所以现在有56天的数据，分成8*7天,剔除部分特征，并加一部分差分特征和离散预测特征（0.386）
predictors_8 = predictors_3+[i+'_median_diff' for i in coll]+[i+'_avg_diff' for i in coll]+[str(i)+'_dis' for i in  na_col_filter]
test_data = data_offline_filter_merge.loc[data_offline_filter_merge.loc[:,'date'].isin(date_sorted[56:]),:]
data_online_proba_df_dic={}
df_fi_dic = {}
data_test_proba_dic = {}
score_dic = {}
dep = 'label'

for i in range(0,56,7):
    train_data = data_offline_filter_merge.loc[data_offline_filter_merge.loc[:,'date'].isin(date_sorted[i:i+7]),:]
    data_online_proba_df,df_fi,data_test_proba,score = model_predict(train_data,test_data,predictors_8,dep,data_online_merge)
    data_online_proba_df_dic[i] = data_online_proba_df
    df_fi_dic[i] = df_fi
    data_test_proba_dic[i] = data_test_proba
    score_dic[i] = score


k = 0
j = 0    
for i in range(0,56,7):
    k = k+data_test_proba_dic[i]
    j = j+data_online_proba_df_dic[i]['score']
    
ant_score(test_data[dep],k) #0.476
false_positive_rate, recall, thresholds = roc_curve(test_data[dep],k)
auc(false_positive_rate, recall)#0.981

data_online['score'] = j
data_online['score'] = data_online['score'].apply(lambda x:decimal.Decimal(float(x/8)).quantize(decimal.Decimal('1.00000000000'), decimal.ROUND_DOWN))
data_online.loc[:,['id','score']].to_csv('data/data_online_proba_df_0605.csv',encoding='utf8',index=False)


'''
整合上面两部分dis的数据

'''
data_offline_filter_na_pred = pd.read_csv('data/data_offline_filter_na_pred.csv')
data_online_na_pred = pd.read_csv('data/data_online_na_pred.csv')
data_offline_filter_na_pred_filter = pd.read_csv('data/data_offline_filter_na_pred_filter.csv')
data_online_na_pred_filter = pd.read_csv('data/data_online_na_pred_filter.csv')

na_pred_dis_col = data_offline_filter_na_pred.columns[data_offline_filter_na_pred.columns.str.contains('_dis')].tolist()
na_pred_filter_dis_col = data_offline_filter_na_pred_filter.columns[data_offline_filter_na_pred_filter.columns.str.contains('_dis')].tolist()

diff_dis_col = [i for i in na_pred_dis_col if i not in na_pred_filter_dis_col]

data_offline_filter = data_offline_filter_na_pred_filter.merge(data_offline_filter_na_pred.loc[:,diff_dis_col+['id']],on='id')
data_online = data_online_na_pred_filter.merge(data_online_na_pred.loc[:,diff_dis_col+['id']],on='id')
na_dis_col = data_offline_filter.columns[data_offline_filter.columns.str.contains('_dis')].tolist()


norm_col = [21,23,25,27,33,35,49,51,57,58,61,62,63,72,73,74,75,80,81,90,91,98,99,100,101,104,105,106,109,110,161,165,177,208,209,210]+c+d
coll = ['f'+str(i) for i in norm_col]

#merge
data_offline_filter_avg_diff = avg_diff(data_offline_filter,coll)
data_offline_filter_med_diff = med_diff(data_offline_filter,coll)
data_offline_filter_merge = data_offline_filter.merge(data_offline_filter_avg_diff).merge(data_offline_filter_med_diff)

data_online_avg_diff = avg_diff(data_online,coll)
data_online_med_diff = med_diff(data_online,coll)
data_online_merge = data_online.merge(data_online_avg_diff).merge(data_online_med_diff)

#将最后六天作为test，所以现在有56天的数据，分成8*7天,剔除部分特征，并加一部分差分特征和离散预测特征（0.4009）
predictors_9 = predictors_3+[i+'_median_diff' for i in coll]+[i+'_avg_diff' for i in coll]+na_dis_col
test_data = data_offline_filter_merge.loc[data_offline_filter_merge.loc[:,'date'].isin(date_sorted[56:]),:]
data_online_proba_df_dic={}
df_fi_dic = {}
data_test_proba_dic = {}
score_dic = {}
dep = 'label'

for i in range(0,56,7):
    train_data = data_offline_filter_merge.loc[data_offline_filter_merge.loc[:,'date'].isin(date_sorted[i:i+7]),:]
    data_online_proba_df,df_fi,data_test_proba,score = model_predict(train_data,test_data,predictors_9,dep,data_online_merge)
    data_online_proba_df_dic[i] = data_online_proba_df
    df_fi_dic[i] = df_fi
    data_test_proba_dic[i] = data_test_proba
    score_dic[i] = score
# {0: [0.9698009440262454, 0.40648967551622417], 7: [0.9660045887765099, 0.38458702064896755], 14: [0.9762243635188219, 0.4002949852507375], 
# 21: [0.9787828871772845, 0.4192477876106195], 28: [0.9791753071372176, 0.4117994100294986], 35: [0.978262483657038, 0.40648967551622417], 
# 42: [0.9798017177368329, 0.45353982300884954], 49: [0.9820597193204941, 0.46297935103244836]}


k = 0
j = 0    
for i in range(0,56,7):
    k = k+data_test_proba_dic[i]
    j = j+data_online_proba_df_dic[i]['score']
    
ant_score(test_data[dep],k) #0.477
false_positive_rate, recall, thresholds = roc_curve(test_data[dep],k)
auc(false_positive_rate, recall)#0.981

data_online['score'] = j
data_online['score'] = data_online['score'].apply(lambda x:decimal.Decimal(float(x/8)).quantize(decimal.Decimal('1.00000000000'), decimal.ROUND_DOWN))
data_online.loc[:,['id','score']].to_csv('data/data_online_proba_df_0606.csv',encoding='utf8',index=False)


'''
在上面的基础上加上无标签的数据，无标签数据全部当成bad放进去（0.35）
'''
na_col = ['f'+str(i) for i in  range(64,155)]
na_col_filter = load_obj('na_col_filter')

predictors_na = [i for i in predictors if i not in  na_col]

#缺失标签的数据,
data_offline_nolabel = data_offline_filter = data_offline.loc[data_offline.loc[:,'label']==-1,:]
data_offline_nolabel[dep] = 1

for coc in na_col_filter: 
    col_offline_nolabel = model_na_pred(data_offline_nolabel,coc,predictors_na)
    data_offline_nolabel = pd.merge(data_offline_nolabel,col_offline_nolabel,on='id')

###
data_offline_filter_na_pred = pd.read_csv('data/data_offline_filter_na_pred.csv')
data_online_na_pred = pd.read_csv('data/data_online_na_pred.csv')
data_offline_filter_na_pred_filter = pd.read_csv('data/data_offline_filter_na_pred_filter.csv')
data_online_na_pred_filter = pd.read_csv('data/data_online_na_pred_filter.csv')

na_pred_dis_col = data_offline_filter_na_pred.columns[data_offline_filter_na_pred.columns.str.contains('_dis')].tolist()
na_pred_filter_dis_col = data_offline_filter_na_pred_filter.columns[data_offline_filter_na_pred_filter.columns.str.contains('_dis')].tolist()

diff_dis_col = [i for i in na_pred_dis_col if i not in na_pred_filter_dis_col]

data_offline_filter = data_offline_filter_na_pred_filter.merge(data_offline_filter_na_pred.loc[:,diff_dis_col+['id']],on='id')
data_online = data_online_na_pred_filter.merge(data_online_na_pred.loc[:,diff_dis_col+['id']],on='id')
na_dis_col = data_offline_filter.columns[data_offline_filter.columns.str.contains('_dis')].tolist()

#将无标签的数据加入
data_offline_filter_ = pd.concat([data_offline_filter,data_offline_nolabel],join='outer')
data_offline_filter_.loc[:,diff_dis_col] = data_offline_filter_.loc[:,diff_dis_col].fillna(data_offline_filter_[diff_dis_col].mode().loc[0,:].to_dict(),axis=0)

#
norm_col = [21,23,25,27,33,35,49,51,57,58,61,62,63,72,73,74,75,80,81,90,91,98,99,100,101,104,105,106,109,110,161,165,177,208,209,210]+c+d
coll = ['f'+str(i) for i in norm_col]
   
#merge
data_offline_filter_avg_diff = avg_diff(data_offline_filter_,coll)
data_offline_filter_med_diff = med_diff(data_offline_filter_,coll)
data_offline_filter_merge = data_offline_filter_.merge(data_offline_filter_avg_diff).merge(data_offline_filter_med_diff)

data_online_avg_diff = avg_diff(data_online,coll)
data_online_med_diff = med_diff(data_online,coll)
data_online_merge = data_online.merge(data_online_avg_diff).merge(data_online_med_diff)

#将最后六天作为test，所以现在有56天的数据，分成8*7天,剔除部分特征，并加一部分差分特征和离散预测特征,测试集中剔除无标签的数据
predictors_9 = predictors_3+[i+'_median_diff' for i in coll]+[i+'_avg_diff' for i in coll]+na_dis_col
test_data = data_offline_filter_merge.loc[data_offline_filter_merge.loc[:,'date'].isin(date_sorted[56:]),:]
test_data = test_data.loc[test_data['id'].isin(set(data_offline_filter.id.tolist())),:]
data_online_proba_df_dic={}
df_fi_dic = {}
data_test_proba_dic = {}
score_dic = {}
dep = 'label'

for i in range(0,56,7):
    train_data = data_offline_filter_merge.loc[data_offline_filter_merge.loc[:,'date'].isin(date_sorted[i:i+7]),:]
    data_online_proba_df,df_fi,data_test_proba,score = model_predict(train_data,test_data,predictors_9,dep,data_online_merge)
    data_online_proba_df_dic[i] = data_online_proba_df
    df_fi_dic[i] = df_fi
    data_test_proba_dic[i] = data_test_proba
    score_dic[i] = score
# {0: [0.9558676331231447, 0.39837758112094396], 7: [0.9619950492455276, 0.28606194690265485], 14: [0.9764020490673679, 0.3998525073746313], 
# 21: [0.9781468672757804, 0.40744837758112096], 28: [0.9798271669607851, 0.4158554572271387], 35: [0.9781671544158679, 0.4044985250737463], 
# 42: [0.980023924377488, 0.46533923303834807], 49: [0.9818503840274351, 0.460693215339233]}


k = 0
j = 0    
for i in range(0,56,7):
    k = k+data_test_proba_dic[i]
    j = j+data_online_proba_df_dic[i]['score']

ant_score(test_data[dep],k) #0.477
false_positive_rate, recall, thresholds = roc_curve(test_data[dep],k)
auc(false_positive_rate, recall)#0.981

data_online['score'] = j
data_online['score'] = data_online['score'].apply(lambda x:decimal.Decimal(float(x/8)).quantize(decimal.Decimal('1.00000000000'), decimal.ROUND_DOWN))
data_online.loc[:,['id','score']].to_csv('data/data_online_proba_df_0625.csv',encoding='utf8',index=False)


'''
开始B榜,先不加无标签的数据（0.3438）
'''
na_col = ['f'+str(i) for i in  range(64,155)]
na_col_filter = load_obj('na_col_filter')

predictors_na = [i for i in predictors if i not in  na_col]

#B榜的数据,

# for coc in na_col_filter: 
    # col_online_b = model_na_pred(data_online_b,coc,predictors_na)
    # data_online_b = pd.merge(data_online_b,col_online_b,on='id')

#data_online_b.to_csv('data/data_online_b_na_pred_filter.csv',index=False)

###
data_offline_filter_na_pred_filter = pd.read_csv('data/data_offline_filter_na_pred_filter.csv')
data_online_b_na_pred_filter = pd.read_csv('data/data_online_b_na_pred_filter.csv')

#na_pred_dis_col = data_offline_filter_na_pred.columns[data_offline_filter_na_pred.columns.str.contains('_dis')].tolist()
na_pred_filter_dis_col = data_offline_filter_na_pred_filter.columns[data_offline_filter_na_pred_filter.columns.str.contains('_dis')].tolist()

#diff_dis_col = [i for i in na_pred_dis_col if i not in na_pred_filter_dis_col]

data_offline_filter = data_offline_filter_na_pred_filter#.merge(data_offline_filter_na_pred.loc[:,diff_dis_col+['id']],on='id')
data_online = data_online_b_na_pred_filter#.merge(data_online_na_pred.loc[:,diff_dis_col+['id']],on='id')
#na_dis_col = data_offline_filter.columns[data_offline_filter.columns.str.contains('_dis')].tolist()
na_dis_col = na_pred_filter_dis_col

#将无标签的数据加入
#data_offline_filter_ = pd.concat([data_offline_filter,data_offline_nolabel],join='outer')
#data_offline_filter_.loc[:,diff_dis_col] = data_offline_filter_.loc[:,diff_dis_col].fillna(data_offline_filter_[diff_dis_col].mode().loc[0,:].to_dict(),axis=0)

norm_col = [21,23,25,27,33,35,49,51,57,58,61,62,63,72,73,74,75,80,81,90,91,98,99,100,101,104,105,106,109,110,161,165,177,208,209,210]+c+d
coll = ['f'+str(i) for i in norm_col]
   
#merge
data_offline_filter_avg_diff = avg_diff(data_offline_filter,coll)
data_offline_filter_med_diff = med_diff(data_offline_filter,coll)
data_offline_filter_merge = data_offline_filter.merge(data_offline_filter_avg_diff).merge(data_offline_filter_med_diff)

data_online_avg_diff = avg_diff(data_online,coll)
data_online_med_diff = med_diff(data_online,coll)
data_online_merge = data_online.merge(data_online_avg_diff).merge(data_online_med_diff)

#将最后六天作为test，所以现在有56天的数据，分成8*7天,剔除部分特征，并加一部分差分特征和（不加离散预测特征）
predictors_9 = predictors_3+[i+'_median_diff' for i in coll]+[i+'_avg_diff' for i in coll]+na_dis_col
test_data = data_offline_filter_merge.loc[data_offline_filter_merge.loc[:,'date'].isin(date_sorted[56:]),:]
data_online_proba_df_dic={}
df_fi_dic = {}
data_test_proba_dic = {}
score_dic = {}
dep = 'label'

for i in range(0,56,7):
    train_data = data_offline_filter_merge.loc[data_offline_filter_merge.loc[:,'date'].isin(date_sorted[i:i+7]),:]
    data_online_proba_df,df_fi,data_test_proba,score = model_predict(train_data,test_data,predictors_9,dep,data_online_merge)
    data_online_proba_df_dic[i] = data_online_proba_df
    df_fi_dic[i] = df_fi
    data_test_proba_dic[i] = data_test_proba
    score_dic[i] = score

# {0: [0.9726842914356635, 0.3938053097345133], 7: [0.9616452428146427, 0.29564896755162245], 14: [0.975651274386068, 0.4064159292035398], 
# 21: [0.9786744721287655, 0.42507374631268435], 28: [0.9799755693469981, 0.4331120943952802], 35: [0.9778317092634905, 0.40324483775811204], 
# 42: [0.9797726114097345, 0.4519174041297935], 49: [0.9815061648374679, 0.4573746312684366]}

k = 0
j = 0    
for i in range(0,56,7):
    k = k+data_test_proba_dic[i]
    j = j+data_online_proba_df_dic[i]['score']

ant_score(test_data[dep],k) #475
false_positive_rate, recall, thresholds = roc_curve(test_data[dep],k)
auc(false_positive_rate, recall) #981

data_online['score'] = j
data_online['score'] = data_online['score'].apply(lambda x:decimal.Decimal(float(x/8)).quantize(decimal.Decimal('1.00000000000'), decimal.ROUND_DOWN))
data_online.loc[:,['id','score']].to_csv('data/data_online_proba_df_0701.csv',encoding='utf8',index=False)



'''
B榜，不加无标签的数据，重新按缺失率剔除需要填充缺失值部分的高线性相关的变量，并预测填充缺失值(0.3)
'''
#手动剔除变量
#a = [str(i) for i in range(20,25)]
b = ['20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','46','47','48','49','50','51','52','53']
c = [str(i) for i in range(64,72)]
d = [str(i) for i in range(111,155)]

drop_col = ['f'+i for i in (b+c+d)]
predictors_3 = [i for i in predictors if i not in drop_col]

#需要差值的变量
norm_col = [21,23,25,27,33,35,49,51,57,58,61,62,63,72,73,74,75,80,81,90,91,98,99,100,101,104,105,106,109,110,161,165,177,208,209,210]+c+d
coll = ['f'+str(i) for i in norm_col]

#剔除需要填充缺失值部分的高线性相关的变量
na_col = ['f'+str(i) for i in  range(64,161)]
na_col_1 = ['f'+str(i) for i in  range(64,72)]+['f'+str(i) for i in  range(76,102)]+['f'+str(i) for i in  range(111,155)] #181597
na_col_2 = ['f'+str(i) for i in  range(72,76)]+['f'+str(i) for i in  range(107,111)]+['f'+str(i) for i in  range(155,161)] #181008
na_col_3 = ['f'+str(i) for i in  range(102,107)] #175191

train_col = data_offline.loc[data_offline.loc[:,'date'].isin(date_sorted[:46]),:]

df_corr_na_col_1 = train_col.fillna(-1).loc[:,na_col_1].corr().applymap(lambda x: abs(x))
corr_high_var_1 = corr_filter(df_corr_na_col_1,0.98)
na_col_1_filter = [i for i in na_col_1 if i not in corr_high_var_1]

df_corr_na_col_2 = train_col.fillna(-1).loc[:,na_col_2].corr().applymap(lambda x: abs(x))
corr_high_var_2 = corr_filter(df_corr_na_col_2,0.98)
na_col_2_filter = [i for i in na_col_2 if i not in corr_high_var_2]

df_corr_na_col_3 = train_col.fillna(-1).loc[:,na_col_3].corr().applymap(lambda x: abs(x))
corr_high_var_3 = corr_filter(df_corr_na_col_3,0.98)
na_col_3_filter = [i for i in na_col_3 if i not in corr_high_var_3]

na_col_filter = na_col_2_filter+na_col_3_filter+na_col_1_filter

save_obj(na_col_filter,'na_col_filter_2')

#填充缺失，训练或直接预测
import copy 
predictors_na = [i for i in predictors if i not in  na_col] 
data_offline_filter_cp = copy.deepcopy(data_offline_filter)
data_online_cp = copy.deepcopy(data_online)

    
# for coc in na_col_filter:
    # print(coc)
    # try:
        # col_data_offline_filter = model_na_pred(data_offline_filter,coc,predictors_na)
        # data_offline_filter = pd.merge(data_offline_filter,col_data_offline_filter,on='id')
        # col_data_online = model_na_pred(data_online,coc,predictors_na)
        # data_online = pd.merge(data_online,col_data_online,on='id')
    # except:
        # col_offline,col_online = model_na_train(data_offline_filter_cp,data_online_cp,coc,predictors_na)
        # data_offline_filter = pd.merge(data_offline_filter,col_offline,on='id')
        # data_online = pd.merge(data_online,col_online,on='id')
na_col_pred_list = []
for coc in na_col:
    print(coc)
    try:
        col_data_offline_filter = model_na_pred(data_offline_filter,coc,predictors_na)
        data_offline_filter = pd.merge(data_offline_filter,col_data_offline_filter,on='id')
        col_data_online = model_na_pred(data_online,coc,predictors_na)
        data_online = pd.merge(data_online,col_data_online,on='id')
        na_col_pred_list.append(coc)
    except:
        pass

save_obj(na_col_pred_list,'na_col_pred_list')

#差分
data_offline_filter_avg_diff = avg_diff(data_offline_filter,coll)
data_offline_filter_med_diff = med_diff(data_offline_filter,coll)
data_offline_filter_merge = data_offline_filter.merge(data_offline_filter_avg_diff).merge(data_offline_filter_med_diff)

data_online_avg_diff = avg_diff(data_online,coll)
data_online_med_diff = med_diff(data_online,coll)
data_online_merge = data_online.merge(data_online_avg_diff).merge(data_online_med_diff)


#将最后六天作为test，所以现在有56天的数据，分成8*7天,剔除部分特征，并加一部分差分特征和离散预测特征（）
predictors_8 = predictors_3+[i+'_median_diff' for i in coll]+[i+'_avg_diff' for i in coll]+[str(i)+'_dis' for i in  na_col_pred_list]
test_data = data_offline_filter_merge.loc[data_offline_filter_merge.loc[:,'date'].isin(date_sorted[56:]),:]
data_online_proba_df_dic={}
df_fi_dic = {}
data_test_proba_dic = {}
score_dic = {}
dep = 'label'

for i in range(0,56,7):
    train_data = data_offline_filter_merge.loc[data_offline_filter_merge.loc[:,'date'].isin(date_sorted[i:i+7]),:]
    data_online_proba_df,df_fi,data_test_proba,score = model_predict(train_data,test_data,predictors_8,dep,data_online_merge)
    data_online_proba_df_dic[i] = data_online_proba_df
    df_fi_dic[i] = df_fi
    data_test_proba_dic[i] = data_test_proba
    score_dic[i] = score

# {0: [0.9584681081191393, 0.3657817109144542], 7: [0.9585576619968019, 0.35265486725663714], 
# 14: [0.9679775507745128, 0.3532448377581121], 21: [0.9678872255942318, 0.3662979351032448], 
# 28: [0.9685247391890989, 0.3471238938053097], 35: [0.9687982242852899, 0.38458702064896755], 
# 42: [0.9720395652207022, 0.4019174041297935], 49: [0.9764130843384874, 0.4346607669616519]}

k = 0
j = 0    
for i in range(0,56,7):
    k = k+data_test_proba_dic[i]
    j = j+data_online_proba_df_dic[i]['score']
    
ant_score(test_data[dep],k) #43
false_positive_rate, recall, thresholds = roc_curve(test_data[dep],k)
auc(false_positive_rate, recall)#975

data_online['score'] = j
data_online['score'] = data_online['score'].apply(lambda x:decimal.Decimal(float(x/8)).quantize(decimal.Decimal('1.00000000000'), decimal.ROUND_DOWN))
data_online.loc[:,['id','score']].to_csv('data/data_online_proba_df_0705.csv',encoding='utf8',index=False)