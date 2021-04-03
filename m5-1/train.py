import pandas as pd
import numpy as np
import gc
import lightgbm  as lgb

from sklearn.model_selection import train_test_split

"""
lightgbmの学習を行う
    train_dataset：学習用データを準備する関数
    train：lightgbmを学習させる関数
"""

#=============================
# train_dataset
#=============================
def train_dataset(data_folder):
    train = pd.read_csv(data_folder + '/train.csv')
    train = train.iloc[:,1:]
    drop_list = ['demand', 'day']
    X = train.drop(drop_list, axis = 1).values
    y = train['demand'].values

    x_train, x_val, y_train, y_val = train_test_split(X, y, random_state = 71, shuffle = True)

    del X, y, train
    gc.collect

    return x_train, x_val, y_train, y_val

#================================
# train
#================================
def train(seed, learning_rate, max_depth, num_leaves, min_child_weight, reg_alpha, subsample, num_boost_round, early_stopping_rounds, data_folder):
    x_train, x_val, y_train, y_val = train_dataset(data_folder)

    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_val, y_val, reference = lgb_train)

    lgbm_params = {
        'boosting_type':'gbdt',
        'objective':'regression',
        'learning_rate':learning_rate,
        'max_depth':max_depth,
        'num_leaves':num_leaves,
        'min_child_weight':min_child_weight,
        'reg_alpha':reg_alpha,
        'subsample':subsample,
        'random_state':seed,
        'verbose':-1
    }

    model = lgb.train(
        lgbm_params, 
        lgb_train, 
        valid_sets = lgb_eval, 
        num_boost_round = num_boost_round, 
        early_stopping_rounds = early_stopping_rounds,
        verbose_eval  = 50
    )

    return model
