import pandas as pd
import gc
import xgboost as xgb
from dateutil.parser import parse
import numpy as np
from sklearn.model_selection import KFold
import sys
import os
from sklearn.metrics import roc_curve, auc

os.chdir('/mnt/Data/xiongwenwen/ant_risk/')
sys.path.append('/mnt/Data/xiongwenwen/py_code')

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

def ant_score(y_true, predict_proba):
    '''
    y_true: numpy.ndarray,不能是带索引的series
    '''
    fpr, tpr, thresholds = roc_curve(y_true, predict_proba, pos_label=1)
    score=0.4*tpr[np.where(fpr>=0.001)[0][0]]+0.3*tpr[np.where(fpr>=0.005)[0][0]]+0.3*tpr[np.where(fpr>=0.01)[0][0]]
    return score


train_data = pd.read_csv('data/train_data.csv')
train_data['ndays'] = train_data['date'].apply(lambda x:(parse(str(x))-parse(str(20170905))).days)
pre = [k for k in train_data.columns if k not in ['id', 'label', 'date', 'bs', 'ndays']]
need_del = ['f20','f21','f22','f23','f24','f25','f26','f27','f32','f33','f34','f35','f48','f49','f50','f51','f52','f53','f64','f65','f66','f67','f68','f69','f70','f71','f111','f112','f113','f114','f115','f116','f117','f118','f119','f120','f121','f122','f123','f124','f125','f126','f127','f128','f129','f130','f131','f132','f133','f134','f135','f136','f137','f138','f139','f140','f141','f142','f143','f144','f145','f146','f147','f148','f149','f150','f151','f152','f153','f154','f161','f162','f163','f164','f165','f211','f212','f213','f214','f215','f216','f217','f218','f219','f220','f221','f222','f223','f224','f225','f226','f227','f228','f229','f230','f231','f232','f233']
pred = [k for k in pre if  k not in need_del]

train = train_data[(((train_data.f28<=1)|(train_data.f28.isnull()))&((train_data.f29<=1)|(train_data.f29.isnull()))&((train_data.f30<=1)|(train_data.f30.isnull()))&((train_data.f31<=1)|(train_data.f31.isnull()))&(train_data['ndays']<=40))]
train['label'] = train['label'].apply(lambda x: 0 if x == 0 else 1)

test = train_data[(((train_data.f28<=1)|(train_data.f28.isnull()))&((train_data.f29<=1)|(train_data.f29.isnull()))&((train_data.f30<=1)|(train_data.f30.isnull()))&((train_data.f31<=1)|(train_data.f31.isnull()))&(train_data['ndays']>40))]
test['label'] = test['label'].apply(lambda x: 0 if x == 0 else 1)
test_dm = xgb.DMatrix(np.array(test[pred]), label=np.array(test['label']))

X = np.array(train[pred])
y = np.array(train['label'])

test_data = pd.read_csv('data/test_b.csv')
dtestt = xgb.DMatrix(np.array(test_data[pred]))
test_data['score'] = 0


##简单bag
kf = KFold(n_splits=5, shuffle=True, random_state=999)
X = np.array(train[pred])
y = np.array(train['label'])

res = []
i = 0
for trn, tst in kf.split(X):
    i+= 1
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
              'nthread': 16,
              'silent': 1,
              'eval_metric': 'auc'
              }
    watchlist = [(dtrain, 'train'), (dtest, 'test')]
    model = xgb.train(params, dtrain, num_boost_round=50000, evals=watchlist,
                      early_stopping_rounds=50)
    test_data['res_s'] = model.predict(dtestt)
    test_data['score'] = test_data['score'] + test_data['res_s'].rank()
    tt_score = ant_score(test['label'], model.predict(test_dm))
    print ('tt_score: ',tt_score)
    res.append(tt_score)
#[0.3294365455502896, 0.3496840442338073, 0.3436545550289626, 0.3368615060558189, 0.3398894154818325]
##


max_v = test_data['score'].max()
min_v = test_data['score'].min()
test_data['score'] = test_data['score'].apply(lambda x: (x - min_v) / (max_v - min_v))

test_data[['id', 'score']].to_csv('data/qjx_data_online_proba_df_0704_2.csv', index=False)#0.4183

#