import pandas as pd
import numpy as np
import lightgbm  as lgb

import warnings
warnings.filterwarnings('ignore')

from train import train

import argparse

"""
submission；submissionファイルの作成を行う関数
"""

parser = argparse.ArgumentParser(
    description = "parameter for lightgbm"
)

parser.add_argument("--seed", default = 71, type = int,
                    help = "lightgbmのパラメータ seed")
parser.add_argument('--learning_rate', default = 0.3, type = float,
                    help = "lightgbmのパラメータ learning_rate")
parser.add_argument('--max_depth', default = 5, type = int,
                    help = "lightgbmのパラメータ max_depth")
parser.add_argument('--num_leaves', default = 31, type = int,
                    help = 'lightgbmのパラメータ num_leaves')
parser.add_argument('--min_child_weight', default = 1, type = int,
                    help = "lightgbmのパラメータ min_child_weight")
parser.add_argument('--reg_alpha', default = 0.0, type = float,
                    help = "lightgbmのパラメータ reg_alpha")
parser.add_argument('--subsample', default = 0.8, type = float,
                    help = "lightgbmのパラメータ subsample")
parser.add_argument('--num_boost_round', default = 500, type = int,
                    help = "lightgbmのパラメータ learning_rate")
parser.add_argument('--early_stopping_rounds', default = 20, type = int,
                    help = "lightgbmのパラメータ learning_rate")
parser.add_argument('--data_folder', type = str,
                    help = "データフォルダへのパス")
parser.add_argument('--output_folder', default = "/kaggle/working", type = str,
                    help = "submissionファイルをoutputしたいフォルダ")

args = parser.parse_args()

#===========================
# submission
#===========================
def submission():
    #モデルの学習
    model = train(
        seed = args.seed,
        learning_rate = args.learning_rate,
        max_depth = args.max_depth,
        num_leaves = args.num_leaves,
        min_child_weight = args.min_child_weight,
        reg_alpha = args.reg_alpha,
        subsample = args.subsample,
        num_boost_round = args.num_boost_round,
        early_stopping_rounds = args.early_stopping_rounds,
        data_folder = args.data_folder
    )

    #推論パート
    df_val = pd.read_csv(args.data_folder + "/val.csv")
    df_eval = pd.read_csv(args.data_folder + '/eval.csv')

    drop_list = ['id','day','demand']

    val_submission = df_val[drop_list]
    eval_submission = df_eval[drop_list]

    x_val = df_val.drop(drop_list, axis = 1)
    x_eval = df_eval.drop(drop_list, axis = 1)

    #予測の出力
    v_pred = model.predict(x_val)
    e_pred = model.predict(x_eval)

    #出力をSeriesに
    val_submission['demand'] = pd.Series(v_pred)
    eval_submission['demand'] = pd.Series(e_pred)

    #逆メルト処理
    val_submission = pd.pivot(val_submission, index = 'id', columns = 'day', values = 'demand').reset_index()
    eval_submission = pd.pivot(eval_submission, index = 'id', columns = 'day', values = 'demand').reset_index()

    #submissionファイルの形に合うように変換
    val_submission.columns = ['id'] + ['F' + str(i+1) for i in range(28)]
    eval_submission.columns = ['id'] + ['F' + str(i+1) for i in range(28)]

    try:
        sub = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
    except:
        sub = pd.read_csv(args.data_folder + '/sample_submission.csv')

    v_submission = pd.merge(sub.iloc[:,:1], val_submission, on = 'id')
    e_submission = pd.merge(sub.iloc[:,:1], eval_submission, on = 'id')

    submission = pd.concat([v_submission,e_submission], axis = 0)

    submission.to_csv(args.output_folder + '/submission.csv', index = False)

if __name__ == '__main__':
    submission()