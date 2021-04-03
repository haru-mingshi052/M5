import pandas as pd

from utility import reduce_mem_usage, encoder_categorical

"""
各データを他データと結合するために加工していく
    pre_sales_train：sales_trainファイルの加工
    pre_calendar：calendarファイルの加工
    pre_sell_prices：sell_pricesの加工
    pre_submission：sample_submissionファイルの加工
"""

#===============================
# pre_sales_train
#===============================
def pre_sales_train(data_folder):
    sales_train = pd.read_csv(data_folder + '/sales_train_evaluation.csv')
    sales_train = reduce_mem_usage(sales_train.iloc[:,:-28]) #容量減らす
    sales_train = encoder_categorical(sales_train, ['item_id','dept_id','cat_id','store_id','state_id'])

    #製品情報"
    product = sales_train.iloc[:,:6]
    product['id'] = product['id'].str.replace('_evaluation', '_validation')

    #売れ行き情報（目的変数）
    demand = sales_train.iloc[:,6:]
    #売れ行き情報を減らす（直近２年分）
    demand = demand.iloc[:,-730:]

    return product, demand

#==================================
# pre_calendar
#==================================
def pre_calendar(data_folder):
    calendar = pd.read_csv(data_folder + '/calendar.csv')
    #必要のない情報の削除
    calendar.drop(['weekday','wday','month','year'], axis=1, inplace=True)
    #カテゴリデータのの数値化
    calendar = encoder_categorical(calendar, ['event_name_1','event_type_1','event_name_2','event_type_2'])

    return calendar

#=======================================
# pre_sell_prices
#=======================================
def pre_sell_prices(data_folder):
    sell_prices = pd.read_csv(data_folder + '/sell_prices.csv')
    #カテゴリデータの数値化
    sell_prices = encoder_categorical(sell_prices,['item_id','store_id'])

    return sell_prices

#========================================
# pre_submission
#========================================
def pre_submission(data_folder):
    valid_submission = pd.read_csv(data_folder + '/sample_submission.csv')
    #submissionファイルのカラム名変更
    valid_submission.columns = ["id"] + [f"d_{d}" for d in range(1914, 1942)]
    #valid用とeval用に分ける
    eval_submission = valid_submission.iloc[30490:]
    valid_submission = valid_submission.iloc[:30490]

    return valid_submission, eval_submission