import pandas as pd
import gc
import xgboost as xgb
from dateutil.parser import parse
import numpy as np
from sklearn.model_selection import KFold

import lightgbm as lgb
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

def tpr_weight_funtion(y_true,y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    a = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    b = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    c = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return 0.4 * a + 0.3 * b + 0.3 * c

train_data = pd.read_csv('or_data\\atec_anti_fraud_train.csv',dtype=creatDtype())
test_data = pd.read_csv('or_data\\atec_anti_fraud_test_b.csv',dtype=creatDtype())
need_del = ['f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','f31','f32','f33','f34','f35','f46','f47','f48','f49','f50','f51','f52','f53','f161','f162','f163','f164','f165','f211','f212','f213','f214','f219','f220','f221','f222','f223']
data = train_data.append(test_data).reset_index(drop=True)
del train_data,test_data
gc.collect()

data['if_join_32'] = data['f32'].apply(lambda x:1 if pd.notnull(x) else 0)
data['if_join_52'] = data['f52'].apply(lambda x:1 if pd.notnull(x) else 0)
data['if_join_161'] = data['f161'].apply(lambda x:1 if pd.notnull(x) else 0)
data['if_join_64'] = data['f64'].apply(lambda x:1 if pd.notnull(x) else 0)

train = data[data.label.notnull()]
test = data[data.label.isnull()]
del data
gc.collect()

train['label'] = train['label'].apply(lambda x: 0 if x == 0 else 1)
test['score'] = 0

pred = [k for k in train.columns if k not in need_del and k not in ['id', 'label', 'date', 'bs', 'ndays'] and k not in ['f'+str(i) for i in range(64,161)]]
kf = KFold(n_splits=5, shuffle=True, random_state=999)
X = np.array(train[pred])
y = np.array(train['label'])

del train
gc.collect()

res = []
i = 0
for trn, tst in kf.split(X):
    i += 1
    print(i)
    trn_X, tst_X = X[trn], X[tst]
    trn_Y, tst_Y = y[trn], y[tst]
    dtrain = lgb.Dataset(trn_X, label=trn_Y)
    dtest = lgb.Dataset(tst_X, label=tst_Y)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'min_child_weight': 1.5,
        'num_leaves': 2 ** 5,
        'lambda_l2': 10,
        'subsample': 0.85,
        'learning_rate': 0.1,
        'seed': 2018,
        'colsample_bytree': 0.5,
        'nthread': 12
    }
    model = lgb.train(params, dtrain, num_boost_round=50000, valid_sets=dtest, early_stopping_rounds=50)
    res.append(tpr_weight_funtion(tst_Y, model.predict(tst_X)))
    del trn_X,trn_Y,tst_X,tst_Y,dtrain,dtest
    gc.collect()
    test['res_s'] = model.predict(np.array(test[pred]))
    test['score'] = test['score'] + test['res_s'].rank()
    del model
    gc.collect()
##

max_v = test['score'].max()
min_v = test['score'].min()
test['score'] = test['score'].apply(lambda x: (x - min_v) / (max_v - min_v))

test[['id', 'score']].to_csv('res_lgb_0703_1.csv', index=False)

#0.4500947930082015
test[['id', 'score','if_join_64']].to_csv('lgb_non_64.csv', index=False)





'''
--------------------------------------------------------------------------------------------------------------------
'''

import pandas as pd
import gc
import xgboost as xgb
from dateutil.parser import parse
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb



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

def tpr_weight_funtion(y_true,y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    a = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    b = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    c = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return 0.4 * a + 0.3 * b + 0.3 * c



train_data = pd.read_csv('or_data\\atec_anti_fraud_train.csv',dtype=creatDtype())
test_data = pd.read_csv('or_data\\atec_anti_fraud_test_b.csv',dtype=creatDtype())
need_del = ['f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','f31','f32','f33','f34','f35','f46','f47','f48','f49','f50','f51','f52','f53','f161','f162','f163','f164','f165','f211','f212','f213','f214','f219','f220','f221','f222','f223']
data = train_data.append(test_data).reset_index(drop=True)
del train_data,test_data
gc.collect()

data['if_join_32'] = data['f32'].apply(lambda x:1 if pd.notnull(x) else 0)
data['if_join_52'] = data['f52'].apply(lambda x:1 if pd.notnull(x) else 0)
data['if_join_161'] = data['f161'].apply(lambda x:1 if pd.notnull(x) else 0)
data['if_join_64'] = data['f64'].apply(lambda x:1 if pd.notnull(x) else 0)

train = data[((data.label.notnull())&(data.if_join_64==1))]
test = data[((data.label.isnull()&(data.if_join_64==1)))]
del data
gc.collect()

train['label'] = train['label'].apply(lambda x: 0 if x == 0 else 1)
test['score'] = 0

pred = [k for k in train.columns if k not in need_del and k not in ['id', 'label', 'date', 'bs', 'ndays']]
kf = KFold(n_splits=5, shuffle=True, random_state=999)
X = np.array(train[pred])
y = np.array(train['label'])

del train
gc.collect()

res = []
i = 0
for trn, tst in kf.split(X):
    i += 1
    print(i)
    trn_X, tst_X = X[trn], X[tst]
    trn_Y, tst_Y = y[trn], y[tst]
    dtrain = lgb.Dataset(trn_X, label=trn_Y)
    dtest = lgb.Dataset(tst_X, label=tst_Y)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'min_child_weight': 1.5,
        'num_leaves': 2 ** 5,
        'lambda_l2': 10,
        'subsample': 0.85,
        'learning_rate': 0.1,
        'seed': 2018,
        'colsample_bytree': 0.5,
        'nthread': 12
    }
    model = lgb.train(params, dtrain, num_boost_round=50000, valid_sets=dtest, early_stopping_rounds=50)
    res.append(tpr_weight_funtion(tst_Y, model.predict(tst_X)))
    del trn_X,trn_Y,tst_X,tst_Y,dtrain,dtest
    gc.collect()
    test['res_s'] = model.predict(np.array(test[pred]))
    test['score'] = test['score'] + test['res_s'].rank()
    del model
    gc.collect()
##

max_v = test['score'].max()
min_v = test['score'].min()
test['score'] = test['score'].apply(lambda x: (x - min_v) / (max_v - min_v))

test[['id', 'score']].to_csv('lgb_64_null_.csv', index=False)


#0.5726168317595552
d1 = pd.read_csv('lgb_non_64.csv')
d2 = pd.read_csv('lgb_64_null_.csv')
d2['rankkk'] = d2['score'].rank(method='first')
d2_dict = dict(zip(d2.rankkk.astype(int),d2.id))


a = d1[d1.if_join_64==1]
a['fz'] = a['score'].rank(method='first')
d1 = d1.merge(a[['id','fz']],how = 'left',on = 'id')
d1['res'] = d1.apply(lambda x:x['id'] if x['if_join_64']==0 else d2_dict[int(x['fz'])],axis =1)

res = pd.DataFrame()
res['id'] = d1['res']
res['score'] = d1['score']

res_ = d1[['id']].merge(res,on = 'id')
res_.to_csv('res_lgb_0703_2.csv',index=False)