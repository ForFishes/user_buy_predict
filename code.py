# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 20:19:08 2018
@author:
"""
import random
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import sys
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import time
import gc
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("ignore")

OFF_LINE = False


def xgb_model(train_set_x, train_set_y, test_set_x):
    """
    xgb训练模型
    :param train_set_x: 训练集样本   type: DataFrame
    :param train_set_y: 训练集标签   type: DataFrame
    :param test_set_x:  测试集样本   type: DataFrame
    :return: predict:   测试集预测结果  type: np.array
    """
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'eta': 0.02,
              'max_depth': 5,  # 4 3
              'colsample_bytree': 0.7,  # 0.8
              'subsample': 0.7,
              'min_child_weight': 9,  # 2 3
              'silent': 1
              }
    dtrain = xgb.DMatrix(train_set_x, label=train_set_y)
    dvali = xgb.DMatrix(test_set_x)
    model = xgb.train(params, dtrain, num_boost_round=800)
    predict = model.predict(dvali)
    return predict

def xgb_CV_model(train_set_x, train_set_y, test_set_x):
    """
    xgb训练模型
    :param train_set_x: 训练集样本   type: DataFrame
    :param train_set_y: 训练集标签   type: DataFrame
    :param test_set_x:  测试集样本   type: DataFrame
    :return: predict:   测试集预测结果  type: np.array
    """
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'eta': 0.02,
              'max_depth': 5,  # 4 3
              'colsample_bytree': 0.7,  # 0.8
              'subsample': 0.7,
              'min_child_weight': 9,  # 2 3
              'silent': 1
              }

    dtrain = xgb.DMatrix(train_set_x, label=train_set_y)
    dvali = xgb.DMatrix(test_set_x)

    # 通过cv找最佳的nround
    cv_log = xgb.cv(params, dtrain, num_boost_round=1000, nfold=5, metrics='auc', early_stopping_rounds=80,
                    seed=2018)
    bst_auc = cv_log['test-auc-mean'].max()
    cv_log['nb'] = cv_log.index
    cv_log.index = cv_log['test-auc-mean']
    nround = cv_log.nb.to_dict()[bst_auc]
    print("最优的迭代次数{}".format(nround))

    watchlist = [(dtrain, 'train')]
    model = xgb.train(params, dtrain, num_boost_round=nround, evals=watchlist)

    # model = xgb.train(params, dtrain, num_boost_round=800)
    predict = model.predict(dvali)
    return predict


def xgb_bagging_model(train_set_x, train_set_y, test_set_x,finall_result):
    """
    xgb训练模型
    :param train_set_x: 训练集样本   type: DataFrame
    :param train_set_y: 训练集标签   type: DataFrame
    :param test_set_x:  测试集样本   type: DataFrame
    :return: predict:   测试集预测结果  type: np.array
    """
    loop_round = 10
    random_seed = list(range(2048))
    max_depth = [4, 5]
    lambd = list(range(50, 150))
    subsample = [i / 1000.0 for i in range(700, 800)]
    colsample_bytree = [i / 1000.0 for i in range(700, 800)]
    min_child_weight = [i / 100.0 for i in range(150, 250)]
    n_feature = [i / 100.0 for i in range(1, 80)]
    # nround = list(range(400,600,5)) # xgboost: the iteration numbers

    random.shuffle(random_seed)
    random.shuffle(max_depth)
    random.shuffle(lambd)
    random.shuffle(subsample)
    random.shuffle(colsample_bytree)
    random.shuffle(min_child_weight)
    random.shuffle(n_feature)
    # random.shuffle(nround)
    # predict_rest = np.array()
    for i in range(loop_round):
        params = {'booster': 'gbtree',
                  'objective': 'binary:logistic',
                  'eval_metric': 'auc',
                  'eta': 0.02,
                  'max_depth': max_depth[i%len(max_depth)],  # 4 3
                  'colsample_bytree': colsample_bytree[i%len(colsample_bytree)],  # 0.8
                  'subsample': subsample[i%len(subsample)],
                  'min_child_weight': min_child_weight[i%len(min_child_weight)],  # 2 3
                  'silent': 1,
                  'seed': random_seed[i%len(random_seed)],
                  'lambda': lambd[i%len(lambd)],
                  }

        dtrain = xgb.DMatrix(train_set_x, label=train_set_y)
        dvali = xgb.DMatrix(test_set_x)

        # 通过cv找最佳的nround
        cv_log = xgb.cv(params, dtrain, num_boost_round=1000, nfold=5, metrics='auc', early_stopping_rounds=80,
                    seed=2018)
        bst_auc = cv_log['test-auc-mean'].max()
        cv_log['nb'] = cv_log.index
        cv_log.index = cv_log['test-auc-mean']
        nround = cv_log.nb.to_dict()[bst_auc]
        print("第{}最优的迭代次数{}".format(i,nround))

        watchlist = [(dtrain, 'train')]
        model = xgb.train(params, dtrain, num_boost_round=nround, evals=watchlist)

        # model = xgb.train(params, dtrain, num_boost_round=800)
        predict = model.predict(dvali)
        # save the file

        result_name = pd.DataFrame()
        # test_result = pd.DataFrame(df_test_x['uid'], columns=["uid"])
        result_name['RST'] = predict
        result_name.to_csv("C:/Users/yuezihan/Desktop/MerchantsBank/nround{0}_xgb{1}.csv".format(nround, i), index=None)

        finall_result['RST'+str(i)] = predict

    finall_result.to_csv('C:/Users/yuezihan/Desktop/MerchantsBank/finall_xgb_result.csv', index=None, sep='\t')
    return


def lgb_model(train_set_x, train_set_y, test_set_x, test_set_y):
    """
    lgb训练模型
    :param train_set_x: 训练集样本    type: DataFrame
    :param train_set_y: 训练集标签    type: DataFrame
    :param test_set_x: 验证集样本     type: DataFrame
    :param test_set_y: 验证集标签     type: np.array
    :return:predict:   测试集预测结果  type: np.array
    """
    lgb_train = lgb.Dataset(train_set_x, train_set_y)
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 32,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    if test_set_y is not None:
        lgb_eval = lgb.Dataset(test_set_x, test_set_y)
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=40000,
                        valid_sets=lgb_eval,
                        verbose_eval=250,
                        early_stopping_rounds=100)
        predict = gbm.predict(test_set_x, num_iteration=gbm.best_iteration)
    else:
        gbm = lgb.train(params,lgb_train,num_boost_round=4000)
        predict = gbm.predict(test_set_x, num_iteration=gbm.best_iteration)

    return predict


def lgb_bagging_model(train_set_x, train_set_y, test_set_x,finall_result):
    """
    xgb训练模型
    :param train_set_x: 训练集样本   type: DataFrame
    :param train_set_y: 训练集标签   type: DataFrame
    :param test_set_x:  测试集样本   type: DataFrame
    :return: predict:   测试集预测结果  type: np.array
    """
    num_leaves = [32]
    feature_fraction = [i / 1000.0 for i in range(800, 950)]
    bagging_fraction = [i / 1000.0 for i in range(800, 900)]

    loop_round = 2
    random.shuffle(num_leaves)
    random.shuffle(feature_fraction)
    random.shuffle(bagging_fraction)

    lgb_train = lgb.Dataset(train_set_x, train_set_y)
    for i in range(loop_round):
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'auc'},
            'num_leaves': num_leaves[i%len(num_leaves)],
            'learning_rate': 0.01,
            'feature_fraction': feature_fraction[i%len(feature_fraction)],
            'bagging_fraction': bagging_fraction[i%len(bagging_fraction)],
            'bagging_freq': 5,
            'verbose': 0
        }
        gbm = lgb.train(params,lgb_train,num_boost_round=4000)
        predict = gbm.predict(test_set_x, num_iteration=gbm.best_iteration)

        result_name = pd.DataFrame()
        # test_result = pd.DataFrame(df_test_x['uid'], columns=["uid"])
        result_name['RST'] = predict
        result_name.to_csv("./lgb/nround{0}_xgb{1}.csv".format(gbm.best_iteration, i), index=None)

        finall_result['RST'+str(i)] = predict

    finall_result.to_csv('finall_lgb_result.csv', index=None, sep='\t')
    return


#获取最近的一次和两次的时间间隔
def getnextday(x):
    if(len(x)) == 1:
        return 32
    x.sort()
    d_1,d_2 = x[0],x[1]
    return d_2-d_1

#获取多种时间间隔的信息
def getmean(x):
    if len(x)<2:
        return 0
    temp = list(set(list(x[0:-1])))
    return np.mean(np.array(temp))
def getmax(x):
    if len(x)<2:
        return -1
    temp = list(set(list(x[0:-1])))
    return np.max(np.array(temp))
def getmin(x):
    if len(x)<2:
        return -1
    temp = list(set(list(x[0:-1])))
    return np.min(np.array(temp))
def getstd(x):
    if len(x)<2:
        return 0
    temp = list(set(list(x[0:-1])))
    return np.std(np.array(temp))


def log_tabel(all_train,data):
    """
    处理log表特征并将其与agg拼接的函数，log表的特征工程可以在该函数中增加或修改
    :param all_train: train_agg与test_agg的拼接   type: DataFrame
    :param data: train_log 与 test_log的拼接   type: DataFrame
    :return: all_train:  从log表中提取特征后与agg表拼接的完整数据集   type:DataFrame
    """
    data['hour'] = data.OCC_TIM.map(lambda x: x[11:13])
    data['day'] = data.OCC_TIM.map(lambda x: x[8:10])




    EVT_LBL_len = data.groupby(by=['USRID'], as_index=False)['EVT_LBL'].agg({'EVT_LBL_len': len})
    EVT_LBL_set_len = data.groupby(by=['USRID'], as_index=False)['EVT_LBL'].agg(
        {'EVT_LBL_set_len': lambda x: len(set(x))})

    TCH_TYP_sum = data.groupby('USRID', as_index=False)['TCH_TYP'].agg({'TCH_TYP': sum})
    TCH_TYP_mean = data.groupby('USRID', as_index = False)['TCH_TYP'].agg({'TCH_TYP': np.mean})
    # hour_len_EVT
    EVT_LBL_hour_len = data.groupby(['USRID', 'hour'])['EVT_LBL'].agg({'EVT_LBL_hour_len': len}).unstack('hour')
    EVT_LBL_hour_len.fillna(0,inplace=True)
    EVT_LBL_hour_len_set = data.groupby(['USRID', 'hour'])['EVT_LBL'].agg({'EVT_LBL_hour_len_set': lambda x: len(set(x))}).unstack('hour')
    EVT_LBL_hour_len_set.fillna(0, inplace=True)
    #day_len_EVT
    EVT_LBL_day_len = data.groupby(['USRID', 'day'])['EVT_LBL'].agg({'EVT_LBL_day_len': len}).unstack('day')
    EVT_LBL_day_len.fillna(0, inplace=True)
    EVT_LBL_day_len_set = data.groupby(['USRID', 'day'])['EVT_LBL'].agg({'EVT_LBL_day_len_set': lambda x: len(set(x))}).unstack('day')
    EVT_LBL_day_len_set.fillna(0, inplace=True)
    #one week len
    EVT_LBL_len['one_week_len'] = EVT_LBL_day_len['EVT_LBL_day_len']['31'].values
    for day in range(1, 8, 1):
        day_str = str(31 - day)
        EVT_LBL_len['one_week_len'] += 1 / float(day + 1) * EVT_LBL_day_len['EVT_LBL_day_len'][day_str].values


    EVT_LBL_set_len['one_week_len_set'] = EVT_LBL_day_len_set['EVT_LBL_day_len_set']['31'].values
    for day in range(1, 8, 1):
        day_str = str(31 - day)
        EVT_LBL_set_len['one_week_len_set'] += 1 / float(day + 1) * EVT_LBL_day_len_set['EVT_LBL_day_len_set'][day_str].values




    #切分EVT_LBL字段
    data['EVT_LBL_1'] = data.EVT_LBL.map(lambda x: x.split('-')[0])
    data['EVT_LBL_2'] = data.EVT_LBL.map(lambda x: x.split('-')[1])
    data['EVT_LBL_3'] = data.EVT_LBL.map(lambda x: x.split('-')[2])
    EVT_LBL_1_set_len = data.groupby(by=['USRID'], as_index = False)['EVT_LBL_1'].agg({'EVT_LBL_1_set_len': lambda x:len(set(x))})
    EVT_LBL_2_set_len = data.groupby(by=['USRID'], as_index=False)['EVT_LBL_2'].agg(
        {'EVT_LBL_2_set_len': lambda x: len(set(x))})
    EVT_LBL_1_hour_len = data.groupby(['USRID', 'hour'])['EVT_LBL_1'].agg({'EVT_LBL_1_hour_len': len}).unstack('hour')
    EVT_LBL_1_hour_set_len =data.groupby(['USRID', 'hour'])['EVT_LBL_1'].agg({'EVT_LBL_1_hour_len_set': lambda x: len(set(x))}).unstack('hour')
    EVT_LBL_1_hour_set_len.fillna(0, inplace=True)
    EVT_LBL_1_len = data.groupby(['USRID', 'EVT_LBL_1']).agg({'EVT_LBL_1': len}).unstack('EVT_LBL_1').fillna(
        0).EVT_LBL_1.reset_index().add_prefix("EVT_LBL_1_").rename(columns={'EVT_LBL_1_USRID':'USRID'})

    EVT_LBL_2_hour_len = data.groupby(['USRID', 'hour'])['EVT_LBL_2'].agg({'EVT_LBL_2_hour_len': len}).unstack('hour')
    EVT_LBL_2_hour_set_len =data.groupby(['USRID', 'hour'])['EVT_LBL_2'].agg({'EVT_LBL_2_hour_len_set': lambda x: len(set(x))}).unstack('hour')
    EVT_LBL_2_hour_set_len.fillna(0, inplace=True)
    EVT_LBL_2_len = data.groupby(['USRID', 'EVT_LBL_2']).agg({'EVT_LBL_2': len}).unstack('EVT_LBL_2').fillna(
        0).EVT_LBL_2.reset_index().add_prefix("EVT_LBL_2_").rename(columns={'EVT_LBL_2_USRID': 'USRID'})

    EVT_LBL_3_len = data.groupby(['USRID', 'EVT_LBL_3']).agg({'EVT_LBL_3': len}).unstack('EVT_LBL_3').fillna(
        0).EVT_LBL_3.reset_index().add_prefix("EVT_LBL_3_").rename(columns={'EVT_LBL_3_USRID': 'USRID'})
    print("EVT_LBL split done...")

    data['OCC_TIM_1'] = data['OCC_TIM'].apply(lambda x: time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S")))
    data = data.sort_values(['USRID', 'OCC_TIM_1'])
    data['next_time'] = data.groupby(['USRID'])['OCC_TIM_1'].diff(-1).apply(np.abs)
    second_feature = data.groupby(['USRID'], as_index=False)['next_time'].agg({
        'next_time_mean': np.mean,
        'next_time_std': np.std,
        'next_time_min': np.min,
        'next_time_max': np.max
    })
    second_feature.fillna(-99,inplace=True)
    #
    #
    #
    #
    #**************20180623**********#增加时间间隔特征
    data['time'] = data.OCC_TIM.apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    now = datetime(2018,4,1,0,0,0)
    data['delta'] = (now-data['time']).apply(lambda x: x.days)
    delta_day_min_max = data.groupby(['USRID'], as_index=False)['delta'].agg({
        'delta_day_min': np.min,
        'delta_day_max': np.max,
    })
    # #****************以上线上0.8664**********#
    #
    #
    #****************点击天数**************
    ClickDay = data.groupby(by=['USRID'], as_index=False)['delta'].mean()
    ClickDay['average_day'] = EVT_LBL_len['EVT_LBL_len'] / ClickDay['delta']

    ClickDay['EVT_LBL_1_average_day'] = EVT_LBL_1_set_len['EVT_LBL_1_set_len'] / ClickDay['delta']
    ClickDay['EVT_LBL_2_average_day'] = EVT_LBL_2_set_len['EVT_LBL_2_set_len'] / ClickDay['delta']
    Delta_days= data.groupby(by=['USRID'], as_index=False)['delta'].agg({'Delta_days': lambda x: getnextday(list(set(x)))})


    print("time feature process done...")
    # #
    # # #***************************拼接特征*********************#
    all_train = pd.merge(all_train, EVT_LBL_len, on=['USRID'], how='left')
    all_train = pd.merge(all_train, EVT_LBL_set_len, on=['USRID'], how='left')
    all_train = pd.merge(all_train, TCH_TYP_sum, on=['USRID'], how='left')
    all_train = pd.merge(all_train, TCH_TYP_mean, on=['USRID'], how='left')
    #**************20180612*******************#
    all_train = pd.merge(all_train, EVT_LBL_hour_len, left_on='USRID', right_index=True, how = 'left')
    all_train = pd.merge(all_train, EVT_LBL_hour_len_set, left_on='USRID', right_index=True, how='left')
    all_train = pd.merge(all_train, EVT_LBL_day_len, left_on='USRID', right_index=True, how='left')
    all_train = pd.merge(all_train, EVT_LBL_day_len_set, left_on='USRID', right_index=True, how='left')
    all_train = pd.merge(all_train, second_feature, on=['USRID'], how = 'left')

    # **************20180618*******************#增加周、小时的信息。
    # all_train = pd.merge(all_train, EVT_LBL_week_day_len, left_on='USRID', right_index=True, how='left')
    # all_train = pd.merge(all_train, df_week_day, on=['USRID'], how='left')
    # all_train = pd.merge(all_train, df_hour_day, on=['USRID'], how='left')
    # # **************20180619*******************#EVT_LBL切分字段信息

    all_train = pd.merge(all_train, EVT_LBL_1_hour_len, left_on='USRID', right_index=True, how='left')
    all_train = pd.merge(all_train, EVT_LBL_1_hour_set_len, left_on='USRID', right_index=True, how='left')
    all_train = pd.merge(all_train, EVT_LBL_2_hour_len, left_on='USRID', right_index=True, how='left')
    all_train = pd.merge(all_train, EVT_LBL_2_hour_set_len, left_on='USRID', right_index=True, how='left')
    all_train = pd.merge(all_train, EVT_LBL_1_len, on='USRID', how = 'left')
    all_train = pd.merge(all_train, EVT_LBL_2_len, on='USRID', how='left')
    #
    #
    #
    # #**********************20180623****时间间隔
    all_train = pd.merge(all_train, delta_day_min_max, on='USRID', how='left')
    all_train = pd.merge(all_train, EVT_LBL_3_len, on='USRID', how='left')
    all_train = pd.merge(all_train, ClickDay, on=['USRID'], how='left')
    all_train = pd.merge(all_train, Delta_days, on=['USRID'], how='left')



    all_train.fillna(-99, inplace=True)

    print("it is over..")
    # all_train.head(100).to_csv('../look.csv')
    return all_train


if __name__ == '__main__':
    train_agg = pd.read_csv('C:/Users/yuezihan/Desktop/MerchantsBank/train/train_agg.csv', sep='\t')
    train_flg = pd.read_csv('C:/Users/yuezihan/Desktop/MerchantsBank/train/train_flg.csv', sep='\t')
    train_log = pd.read_csv('C:/Users/yuezihan/Desktop/MerchantsBank/train/train_log.csv', sep='\t')
    test_agg = pd.read_csv('C:/Users/yuezihan/Desktop/MerchantsBank/test/test_agg.csv', sep='\t')
    test_log = pd.read_csv('C:/Users/yuezihan/Desktop/MerchantsBank/test/test_log.csv', sep='\t')

    # train_agg = train_agg.loc[
    #             ((train_agg.V28 > -10) & (train_agg.V18 < 30) & (train_agg.V20 > -40) & (train_agg.V8 > -5) \
    #              & (train_agg.V19 < 40) & (train_agg.V11 < 60) & (train_agg.V24 > -40) & (train_agg.V9 < 40) \
    #              & (train_agg.V7 > -20) & (train_agg.V7 < 60) & (train_agg.V15 < 50) & (train_agg.V10 < 40) \
    #              & (train_agg.V16 < 100)), :]


    all_agg = pd.concat([train_agg, test_agg],axis=0, copy=False)
    all_log = pd.concat([train_log, test_log],axis=0, copy=False)
    # all_train = pd.merge(train_flg, train_agg, on=['USRID'], how='left')
    all_data = log_tabel(all_agg,all_log)

    #*************加入转化率特征0627*********#
    # train_log_flg = pd.merge(train_log, train_flg, on='USRID', how='left')
    # train_log_flg['day'] = train_log_flg.OCC_TIM.apply(lambda x: x[8:10])
    # train_log_flg['hour'] = train_log_flg.OCC_TIM.apply(lambda x: x[11:13])
    #
    # day_conversion = train_log_flg.groupby('day')['FLAG'].mean()
    # hour_conversion = train_log_flg.groupby('hour')['FLAG'].mean()
    #
    # train_log_flg['day'] = train_log_flg.day.map(day_conversion)
    # train_log_flg['hour'] = train_log_flg.hour.map(hour_conversion)
    #
    # conversion_info = pd.concat([train_log_flg.USRID, train_log_flg.day, train_log_flg.hour],axis=1)
    #
    # all_data = pd.merge(all_data, conversion_info, on='USRID', how='left')
    #******从整体数据集中拆出训练样本与测试样本********
    all_train = all_data.iloc[0:len(train_agg),:].copy()
    all_train = pd.merge(all_train, train_flg, on=['USRID'], how='left')

    test_set = all_data.iloc[len(train_agg):,:].copy()
    del all_data
    del train_agg
    del train_log
    del test_log
    del test_agg
    del train_flg
    gc.collect()


    if OFF_LINE == True:
        train_x = all_train.drop(['USRID', 'FLAG'], axis=1).values
        train_y = all_train['FLAG'].values
        xgb_auc_list = []
        lgb_auc_list = []

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
        for train_index, test_index in skf.split(train_x, train_y):
            print('Train: %s | test: %s' % (train_index, test_index))
            X_train, X_test = train_x[train_index], train_x[test_index]
            y_train, y_test = train_y[train_index], train_y[test_index]

            xgb_pred_value = xgb_model(X_train, y_train, X_test)
            xgb_auc = metrics.roc_auc_score(y_test, xgb_pred_value)
            print('xgboost auc value:', xgb_auc)
            xgb_auc_list.append(xgb_auc)

            lgb_pred_value = lgb_model(X_train, y_train, X_test, y_test)
            # lgb_pred_value = xgb_CV_model(X_train, y_train, X_test)
            lgb_auc = metrics.roc_auc_score(y_test, lgb_pred_value)
            print('lightgbm auc value:', lgb_auc)
            lgb_auc_list.append(lgb_auc)

        print('xgboost validate result:', np.mean(xgb_auc_list))
        print('lightgbm validate result:', np.mean(lgb_auc_list))
        sys.exit(32)


    ###########################
    result_name = pd.DataFrame(test_set.USRID)

    #**************对负样本采样版本********
    # all_train_negative = all_train[all_train.FLAG == 0]
    # all_train_positive = all_train[all_train.FLAG == 1]
    # kf = KFold(n_splits=9)
    # test_x = test_set.drop(['USRID'], axis=1).values
    # pred_result_mean = np.zeros(20000)
    # for train_index, test_index in kf.split(all_train_negative):
    #     train_x_positive = all_train_positive.drop(['USRID', 'FLAG'], axis=1).values
    #     train_x_negative = all_train.iloc[test_index,:].drop(['USRID', 'FLAG'], axis=1).values
    #     train_x = np.concatenate((train_x_positive, train_x_negative), axis=0)
    #     train_y_positive = all_train_positive['FLAG'].values
    #     train_y_negative = all_train.iloc[test_index,:]['FLAG'].values
    #     train_y = np.concatenate((train_y_positive, train_y_negative), axis=0)
    #     pred_result = xgb_model(train_x, train_y, test_x)
    #     pred_result_mean = pred_result_mean + pred_result


    train_x = all_train.drop(['USRID', 'FLAG'], axis=1).values
    train_y = all_train['FLAG'].values
    test_x = test_set.drop(['USRID'], axis=1).values
    pred_result = lgb_bagging_model(train_x, train_y, test_x,result_name)

    # result_name['RST'] = pred_result
    # result_name.to_csv('test_result.csv', index=None, sep='\t')
