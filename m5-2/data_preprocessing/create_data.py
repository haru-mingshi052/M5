import pandas as pd

from preprocessing import pre_sales_train, pre_calendar, pre_sell_prices, pre_submission

import argparse

"""
加工した各データを結合して学習用データを作る
    preprocess_sales：sales_train（加工済み）のデータを変形していく関数
    m5_dataset：加工したそれぞれのファイルをcsvファイルとして出力する関数
"""


parser = argparse.ArgumentParser(
    description = "data augmentation"
)

parser.add_argument("--data_folder", default = "/kaggle/input/m5-forecasting-accuracy", type = str,
                    help = "データのあるフォルダー")
parser.add_argument("--output_folder", default = '/kaggle/working', type = str,
                    help = "加工したデータを出力したいフォルダー")

args = parser.parse_args()


def preprocess_sales(data_folder):
    #product：製品情報  demand：売上情報
    product, demand = pre_sales_train(data_folder)

    valid_submission, eval_submission = pre_submission(data_folder)

    #製品情報と売れ行き情報の結合
    #sales_trainは目的変数の埋まっているtrain-data、sales_validは目的変数の埋まってない提出するデータ
    sales_train = pd.concat([product, demand],axis = 1)
    sales_valid = pd.merge(product, valid_submission, on = 'id')

    product['id'] = product['id'].str.replace('_validation', '_evaluation')
    sales_eval = pd.merge(product, eval_submission, on = 'id')

    #melt変換
    sales_train = pd.melt(
        sales_train,
        id_vars = ['id','item_id','dept_id','cat_id','store_id','state_id'],
        var_name = 'day',
        value_name = 'demand'
    )

    sales_valid = pd.melt(
        sales_valid,
        id_vars = ['id','item_id','dept_id','cat_id','store_id','state_id'],
        var_name = 'day',
        value_name = 'demand'
    )

    sales_eval = pd.melt(
        sales_eval,
        id_vars = ['id','item_id','dept_id','cat_id','store_id','state_id'],
        var_name = 'day',
        value_name = 'demand'
    )

    return sales_train, sales_valid, sales_eval

def m5_dataset():
    sales_train, sales_valid, sales_eval = preprocess_sales(args.data_folder)
    calendar = pre_calendar(args.data_folder)
    sell_prices = pre_sell_prices(args.data_folder)

    #calendarと結合
    train_data = pd.merge(sales_train, calendar, how = 'left', left_on = ['day'], right_on = ['d'])
    valid_data = pd.merge(sales_valid, calendar, how = 'left', left_on = ['day'], right_on = ['d'])
    eval_data = pd.merge(sales_eval, calendar, how = 'left', left_on = ['day'], right_on = ['d'])

    #train_dataとsell_priceを結合
    train_data = train_data.merge(sell_prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')
    valid_data = valid_data.merge(sell_prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')
    eval_data = eval_data.merge(sell_prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')

    #「d」は「day」と被っている、「date」も「day」と似たようなカラムなので削除
    drop_list = ['date', 'd']
    train_data.drop(drop_list, axis=1, inplace = True)
    valid_data.drop(drop_list, axis=1, inplace = True)
    eval_data.drop(drop_list, axis = 1, inplace = True)

    #欠損値があるデータは削除
    train_data.dropna(inplace = True)

    train_data.to_csv(args.output_folder + '/train.csv', index = False)
    valid_data.to_csv(args.output_folder + '/val.csv', index = False)
    eval_data.to_csv(args.output_folder + '/eval.csv', index = False)

if __name__ == '__main__':
    m5_dataset()