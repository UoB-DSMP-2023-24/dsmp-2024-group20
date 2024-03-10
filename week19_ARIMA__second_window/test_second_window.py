import os
import pandas as pd
import numpy as np
from pandas import Timedelta

from sklearn.model_selection import GridSearchCV  # 网格搜索
from matplotlib import pyplot as plt
from pandas import Timedelta
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
import datetime
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm

#%%

"""
using "read_data_LOB.py" to read the LOB data

the folder's path:
path_bid = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/LOB_processed/LOB_bid'
path_ask = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/LOB_processed/LOB_ask'
"""

path_bid_folder = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/LOB_processed/LOB_bid'
path_ask_folder = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/LOB_processed/LOB_ask'

#%%
# read the data

bid_names = os.listdir(path_bid_folder)

## 批量处理
# for bid_name in bid_names:
#     bid_path = os.path.join(path_bid_folder, bid_name)
#     df_bid = pd.read_csv(bid_path)

bid_name = bid_names[0]
bid_path = os.path.join(path_bid_folder, bid_name)
df_bid = pd.read_csv(bid_path)

#%%
## 给数据框加时间-- 由时间戳转化
##错了### df_bid['day'] = pd.to_datetime(df_bid['timestamp'], unit='s')

def wavg(data_csv, timestamp='timestamp', price='price', quantity='quantity'):
    data_csv['wavg_plus_price'] = data_csv[price] * data_csv[quantity]
    df = data_csv.groupby([timestamp]).agg({'wavg_plus_price': 'sum', quantity: 'sum'}).reset_index()
    df['AWVP'] = df['wavg_plus_price'] / df[quantity]
    df['date'] = data_csv['date']
    return df

wavg_bid = wavg(df_bid)

#%%
# aggregation 聚合
def aggregate_data(df, second,
                   aggregation_rules = {'AWVP': 'mean'}):
    df['time_window'] = (df['timestamp'] // second) * second
    aggregated_df = df.groupby('time_window').agg(aggregation_rules).reset_index()
    aggregated_df['date'] = df['date']
    return aggregated_df

agg_bid = aggregate_data(wavg_bid,second=5)

#%%
## 转化时间戳
def timestamp_to_time(data, filename, start_time=[8, 0, 0]):
    # ## 批量处理
    # for filename in filenames:
    data['time'] = pd.to_datetime(data['time_window'], unit='s')
    data['time'] = data['time'].apply(lambda x: x.replace(year=int(filename[:4]),
                                                              month=int(filename[5:7]),
                                                              day=int(filename[8:10])))
    data['time'] += Timedelta(hours=start_time[0], minutes=start_time[1], seconds=start_time[2])
    return data

agg_bid = timestamp_to_time(agg_bid,bid_name)

#%%
# ADF测试 ————平稳性测试
def ADFtest(timeseries):
    # 执行Augmented Dickey-Fuller测试
    print('Results of Augmented Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    # print(dfoutput)
    return dfoutput



#%%
# split the data

def spilt_to_3dataset(df, rate_of_train, rate_of_val):

    train_size = int(len(df) * rate_of_train)
    val_size = train_size + int(len(df)* rate_of_val)

    train = df[:train_size]
    val = df[train_size:val_size]
    test = df[train_size:]
    return train, val, test

#%%
# 模型





#%%
def auto_aggwindow_test(df, second):

    agg_df = aggregate_data(df, second)

    agg_df = timestamp_to_time(agg_bid, bid_name)
    adf_result = ADFtest(agg_df['AWVP'])
    if adf_result['Test Statistic'] >= adf_result['Critical Value (5%)'] or adf_result['p-value'] >= 0.05 :
        # 需要进行差分。 差分限制条件： 统计量不比Critical Value (5%)小，或者 p-value 不比 5% 小
        print('when second = %d, Difference!!!' % second)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
    elif adf_result['Test Statistic'] < adf_result['Critical Value (1%)'] and adf_result['p-value'] < 0.01:
        print("Great! No difference! Let's go!")


    train, val, test = spilt_to_3dataset(agg_df, 0.9, 0.1)
    dataset = train.set_index('time', inplace=False)['AWVP']
    model_auto = pm.auto_arima(dataset, start_p=0, start_q=0,
                               test='adf',  # 使用ADF测试确定'd'
                               max_p=9, max_q=9,  # 设置p和q的最大值
                               m=1,  # 数据的季节性周期
                               d=None,  # 让模型自动确定最优的d
                               seasonal=False,  # 数据不包含季节性成分
                               stepwise=True,  # 使用逐步算法
                               suppress_warnings=True,  # 抑制警告信息
                               information_criterion='aic',  # 使用AIC选择最佳模型
                               trace=False)  # 打印搜索过程
    print(model_auto.summary())
    model_summry = model_auto.summary()
    # print('Got the summary of the model!')
    print('========================================================================')
    return adf_result, model_summry



#%%
for second in range(20):
    adf = []
    model_summ = []

    adf_result, model_summry = auto_aggwindow_test(wavg_bid, second)




