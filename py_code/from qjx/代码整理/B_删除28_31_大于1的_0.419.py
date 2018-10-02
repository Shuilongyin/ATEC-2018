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
train_data['ndays'] = train_data['date'].apply(lambda x:(parse(str(x))-parse(str(20170905))).days)
pre = [k for k in train_data.columns if k not in ['id', 'label', 'date', 'bs', 'ndays']]
need_del = ['f20','f21','f22','f23','f24','f25','f26','f27','f32','f33','f34','f35','f48','f49','f50','f51','f52','f53','f64','f65','f66','f67','f68','f69','f70','f71','f111','f112','f113','f114','f115','f116','f117','f118','f119','f120','f121','f122','f123','f124','f125','f126','f127','f128','f129','f130','f131','f132','f133','f134','f135','f136','f137','f138','f139','f140','f141','f142','f143','f144','f145','f146','f147','f148','f149','f150','f151','f152','f153','f154','f161','f162','f163','f164','f165','f211','f212','f213','f214','f215','f216','f217','f218','f219','f220','f221','f222','f223','f224','f225','f226','f227','f228','f229','f230','f231','f232','f233']
pred = [k for k in pre if  k not in need_del]


train = train_data[(((train_data.f28<=1)|(train_data.f28.isnull()))&((train_data.f29<=1)|(train_data.f29.isnull()))&((train_data.f30<=1)|(train_data.f30.isnull()))&((train_data.f31<=1)|(train_data.f31.isnull()))&(train_data['ndays']<=40))]
train['label'] = train['label'].apply(lambda x: 0 if x == 0 else 1)

test_data = pd.read_csv('or_data\\atec_anti_fraud_test_b.csv',dtype=creatDtype())
dtestt = xgb.DMatrix(np.array(test_data[pred]))
test_data['score'] = 0
del train_data
gc.collect()

res = []
##简单bag
kf = KFold(n_splits=5, shuffle=True, random_state=999)
X = np.array(train[pred])
y = np.array(train['label'])

del train
gc.collect()
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
    test_data['res_s'] = model.predict(np.array(test_data[pred]))
    test_data['score'] = test_data['score'] + test_data['res_s'].rank()
    del model
gc.collect()
##

max_v = test_data['score'].max()
min_v = test_data['score'].min()
test_data['score'] = test_data['score'].apply(lambda x: (x - min_v) / (max_v - min_v))

test_data[['id', 'score']].to_csv('res_lgb_0702_1.csv', index=False)

#0.44540128707138005
