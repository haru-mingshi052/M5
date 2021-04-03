import pandas as pd

from utility import reduce_mem_usage, encoder_categorical

"""
各データを他データと結合するために加工していく関数
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
    calendar.drop(['weekday', 'year'], axis=1, inplace=True)
    calendar = encoder_categorical(calendar, ['event_name_1','event_type_1','event_name_2','event_type_2'])

    #dateの日付の部分を取得
    d_day = pd.Series(calendar['date'].str[-2:], name = 'd_day', dtype = 'int')
    calendar = pd.concat([calendar, d_day], axis = 1)

    #beggining of the year(boty)を作成
    #month == 1かつd_dayがbotyリストの中にある場合、boty特徴量を1に
    #beggining of the year
    calendar['boty'] = 0
    boty_list = [1,2,3,4,5,6,7]
    for i in range (len(calendar)):
        if (calendar.iat[i,3] == 1) & (calendar.iat[i,12] in boty_list):
            calendar.iat[i,13] = 1

    #end of the year(eoty)
    #month == 12かつd_dayがeotyリストにある数字と一緒だった場合eoty特徴量を1に
    #end of the year
    calendar['enty'] = 0
    eoty_list = [25,26,27,28,29,30,31]
    for i in range (len(calendar)):
        if (calendar.iat[i,3] == 12) & (calendar.iat[i,12] in eoty_list):
            calendar.iat[i,13] = 1

    #週末情報
    #曜日の変数を担っているwdayが1か2だった場合、weekend変数を1に
    #weekend
    calendar['weekend'] = 0
    for i in range (len(calendar)):
        if (calendar.iat[i,2] == 1) | (calendar.iat[i,2] == 2):
            calendar.iat[i,14] = 1

    return calendar

#=======================================
# pre_sell_prices
#=======================================
def pre_sell_prices(data_folder):
    sell_prices = pd.read_csv(data_folder + '/sell_prices.csv')
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