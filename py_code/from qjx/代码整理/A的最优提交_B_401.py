import pandas as pd
import gc
import xgboost as xgb
from dateutil.parser import parse
import numpy as np
from sklearn.model_selection import KFold

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
train_data['ndays'] = train_data['date'].apply(lambda x:(parse(str(x))-parse(str(20170905))).days)
pre = [k for k in train_data.columns if k not in ['id', 'label', 'date', 'bs', 'ndays']]
need_smoth = ['f1','f3','f4','f5','f6','f7','f13','f28','f29','f30','f31','f42','f43','f44','f45','f54','f55','f56','f57','f58','f76','f77','f78','f79','f82','f83','f84','f85','f86','f87','f88','f102','f107','f161','f162','f163','f164','f165','f191','f192','f193','f204','f205','f206','f207','f208','f209','f210','f211','f212','f213','f214','f215','f216','f217','f218','f219','f220','f221','f222','f223','f224','f225','f226','f227','f228','f229','f230','f231','f232','f233','f234','f235','f236','f237','f238','f239','f240','f241','f242','f243','f244','f245','f246','f247','f248','f249','f250','f251','f252','f253','f259','f260','f261','f262','f263','f264','f265','f266','f270','f271']
same_distri = ['f2','f8','f9','f10','f11','f12','f14','f15','f16','f17','f18','f19','f36','f37','f38','f39','f40','f41','f46','f47','f59','f60','f61','f62','f63','f72','f73','f74','f75','f80','f81','f89','f90','f91','f92','f93','f94','f95','f96','f97','f98','f99','f100','f101','f102','f103','f104','f105','f106','f108','f109','f110','f155','f156','f157','f158','f159','f160','f166','f167','f168','f169','f170','f171','f172','f173','f174','f175','f176','f177','f178','f179','f180','f181','f182','f183','f184','f185','f186','f187','f188','f189','f190','f194','f195','f196','f197','f198','f199','f200','f201','f202','f203','f254','f255','f256','f257','f258','f267','f268','f269','f272','f273','f274','f275','f276','f277','f278','f279','f280','f281','f282','f283','f284','f285','f286','f287','f288','f289','f290','f291','f292','f293','f294','f295','f296','f297']
pred = [k for k in pre if  k in need_smoth or k in same_distri]

train = train_data[train_data['ndays'] <= 39]
del train_data
gc.collect()
train['label'] = train['label'].apply(lambda x: 0 if x == 0 else 1)
X = np.array(train[pred])
y = np.array(train['label'])
del train
gc.collect()

test_data = pd.read_csv('or_data\\atec_anti_fraud_test_b.csv',dtype=creatDtype())
test_data['score'] = 0
res = test_data[['id','score']]
dtestt = xgb.DMatrix(np.array(test_data[pred]))

del test_data
gc.collect()

kf = KFold(n_splits=5, shuffle=True, random_state=999)

i = 0
for trn, tst in kf.split(X):
    i += 1
    print(i)
    trn_X, tst_X = X[trn], X[tst]
    trn_Y, tst_Y = y[trn], y[tst]
    dtrain = xgb.DMatrix(trn_X, label=trn_Y)
    dtest = xgb.DMatrix(tst_X, label=tst_Y)

    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'max_depth': 4,
              'lambda': 10,
              'min_child_weight': 2,
              'eta': 0.1,
              'seed': 2018,
              'nthread': 3,
              'silent': 1,
              'eval_metric': 'auc'
              }

    watchlist = [(dtrain, 'train'), (dtest, 'test')]
    model = xgb.train(params, dtrain, num_boost_round=50000, evals=watchlist,
                      early_stopping_rounds=50)

    res['res_s'] = model.predict(dtestt)
    res['score'] = res['score'] + res['res_s'].rank()
    del trn_X
    del tst_X
    del trn_Y
    del tst_Y
    del dtrain
    del dtest
    del model
    gc.collect()

##


max_v = res['score'].max()
min_v = res['score'].min()
res['score'] = res['score'].apply(lambda x: (x - min_v) / (max_v - min_v))
res[['id', 'score']].to_csv('res_xgb_bagging_0628_unlabel.csv', index=False)
