"""
修改了 read_csvfile_all3()函数： 增加三个参数start_time=[8, 0, 0], AddtheTime = False, AddtheDate = False, 并且读取的数据集(返回值)增加了一列数据'time'
    :param start_time 市场开始交易的时间: int; structure: start_time= [hour, minute, second]
            The meaning of start_time is the time when the stock market starts trading
            ## default: start_time = [8, 0, 0]
            ## The London Stock Exchange opens at 8:00 a.m. UK time
    :param AddtheTime: control whether to add the time column to the dataframe
    :param AddtheDate: control whether to add the date column to the dataframe
    :return data: type: pd.DataFrame; data['time']: Year-Month-Day Hour:Minute:second (!! the type of second is float)



"""



#%%
# 预加载模块
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from pandas import Timedelta
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
import datetime
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm

##########################
#%%
def read_csvfile_all3(directory, start_time=[8, 0, 0], AddtheTime=False, AddtheDate=False):
    """
    Read csv file
    :param directory 数据文件所在的文件夹路径: str; end with '/'
    :param start_time 市场开始交易的时间: int; structure: start_time= [hour, minute, second]
            The meaning of start_time is the time when the stock market starts trading
            ## default: start_time = [8, 0, 0]
            ## The London Stock Exchange opens at 8:00 a.m. UK time
    :param AddtheTime: control whether to add the time column to the dataframe
    :param AddtheDate: control whether to add the date column to the dataframe
    :return: data: pd.DataFrame; keys = "timestamp", "price", "quantity", "time"
            data['time']: Year-Month-Day Hour:Minute:second (!! the type of second is float)
    """
    dataframes = []
    file_names = os.listdir(directory)
    column_names = ['timestamp', 'price', 'quantity']
    for file_name in file_names:
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory, file_name)
            data = pd.read_csv(file_path, header=None, names=column_names, usecols=[0, 1, 2])
            ##
            if AddtheTime != False:
                time_date = file_name[0][10:20]
                data['time'] = pd.to_datetime(data['timestamp'], unit='s')
                data['time'] = data['time'].apply(lambda x: x.replace(year= int(time_date[:4]), month= int(time_date[5:7]), day= int(time_date[8:])))
                data['time'] += Timedelta(hours= start_time[0], minutes= start_time[1], seconds= start_time[2])
            ##
            if AddtheDate != False:
                time_date = file_names[0][10:20]
                data['date'] = pd.to_datetime([time_date] * len(data))

            dataframes.append(data)
    if dataframes:
        dataframe = pd.concat(dataframes, ignore_index=True)
        print('The type of data is', type(dataframe))
        return dataframe, file_names
    else:
        print('No CSV files found or DataFrame is empty.')
        return pd.DataFrame()
#%%
# test 测试  data_csv: UoB_Set01_2025-01-02tapes.csv
## The London Stock Exchange opens at 8:00 a.m. UK time and closes at 4:30 p.m.
### with the market closed from 12:00 to 12:02 noon.
## 伦敦股票交易市场：8:00开盘,16:30收盘。12:00-12:02是休息时间。每天交易时长为8小时28分钟。
## default: start_time = [8, 0, 0]
directory_trytry = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/trytry/'
data_csv, file_names = read_csvfile_all3(directory_trytry)
# data_csv2 = read_csvfile_all3(directory_trytry, AddtheTime=True, AddtheDate=True)
## keys: timestamp  price  quantity  time

summary_data_csv = data_csv.describe()
summary_data_csv

#%%
##########################
# wavg_tapesprice: weight average price based on quantity
## 计算加权平均价格——按照数量加权
def wavg(data_csv):
    df = pd.DataFrame()
    data_csv['wavg_plus_price'] = data_csv['price'] * data_csv['quantity']
    df = data_csv.groupby(['timestamp']).agg({'wavg_plus_price': 'sum', 'quantity': 'sum'}).reset_index()
    df['weighted_avg_price'] = df['wavg_plus_price'] / df['quantity']
    if 'time' in data_csv.keys():
        df['time'] = data_csv['time']
    if 'date' in data_csv.keys():
        df['date'] = data_csv['date']
    return df

wavg_tapesprice = wavg(data_csv)
# wavg_tapesprice2 = wavg(data_csv2)

# summary_wavg = wavg_tapesprice.describe()
# summary_wavg

#%%
def aggregate_data(df, second_column, aggregation_rules,second):
    df['time_window'] = (df[second_column] // second) * second
    aggregated_df = df.groupby('time_window').agg(aggregation_rules).reset_index()
    return aggregated_df

## 假如输入的数据集 包含了多日的文件： 使用" if filename in file_names: " 循环来 处理聚合计算。
### 并在上述循环内 由filename生成时间，下面是参考代码：
### 【注】：file_names 由函数 read_csvfile_all3()得到。 data是经过聚合计算后的数据集。
# ####
# for filename in file_names:
#         time_date = filename[0][10:20]
#         data['time'] = pd.to_datetime(data['timestamp'], unit='s')
#         data['time'] = data['time'].apply(lambda x: x.replace(year=int(time_date[:4]),
#                                                               month=int(time_date[5:7]),
#                                                               day=int(time_date[8:])))
#         data['time'] += Timedelta(hours=start_time[0], minutes=start_time[1], seconds=start_time[2])

aggregation_rules = {'weighted_avg_price': 'mean'}
agg_wavg = aggregate_data(wavg_tapesprice, second_column='timestamp', aggregation_rules=aggregation_rules, second=5)


#%%
## -------------------------------------------------------------
## Converts a timestamp to a time

### test !!!!!!!!!!!!
filename = ['2025-01-02']
## -------------------------
def timestamp_to_time(data, filenames, start_time=[8, 0, 0]):
    """
    Converts a timestamp to a time
    :param data: pandas dataframe with timestamps 【！】 timestamp column's name : "time_window"
    :param filenames: come from the read_csvfile_all3()
    :param start_time: list; default value [8, 0, 0]
    :return: pandas dataframe with time: YYYY-MM-DD HH:mm:ss.ms
    """
    for filename in filenames:
        data['time'] = pd.to_datetime(data['time_window'], unit='s')
        data['time'] = data['time'].apply(lambda x: x.replace(year=int(filename[:4]),
                                                              month=int(filename[5:7]),
                                                              day=int(filename[8:])))
        data['time'] += Timedelta(hours=start_time[0], minutes=start_time[1], seconds=start_time[2])
    return data

agg_wavg_time = timestamp_to_time(agg_wavg, filenames=filename)

# summary_agg = agg_wavg.describe()
# summary_agg

#%%
##########################
# ADF 测试(ADF(Augmented Dickey-Fuller) 强迪基-福勒检验)——时间序列的平稳性
def ADFtest(timeseries):
    # 执行Augmented Dickey-Fuller测试
    print('Results of Augmented Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
### 使用.dropna()移除由于差分产生的任何NaN值.
### 例如： data_diff.dropna()

# 测试 test
## wavg_tapesprice dataset
adf_result_wavg= ADFtest(wavg_tapesprice['weighted_avg_price'])

## agg_wavg dataset
adf_result_agg = ADFtest(agg_wavg_time['weighted_avg_price'])

#%%
## difference : agg_wavg dataset
agg_wavg_time['price_diff1'] = agg_wavg_time['weighted_avg_price'].diff()
adf_result_agg_diff1 = ADFtest(agg_wavg_time['price_diff1'].dropna())
### 使用.dropna()移除由于差分产生的任何NaN值.
### 例如： data_diff.dropna()
#%%
## split the data:  train, val, test

# df = pd.DataFrame(wavg_tapesprice[['timestamp', 'weighted_avg_price']])
df =pd.DataFrame(agg_wavg_time[['time_window', 'weighted_avg_price', 'price_diff1', 'time']])

train_size = int(len(df)*0.8)
val_size = train_size + int(len(df)*0.1)

train = df[:train_size]
val = df[train_size:val_size]
test = df[val_size:]

## --------------------------------------------------
def set_timeindex(dataset):
    # dataset.set_index('time_window', inplace=True)
    dataset.set_index('time', inplace=True)
    return dataset
# set_timeindex(train)
# set_timeindex(val)
# set_timeindex(test)


## test
# timestamp1 = 24480
# time_ = datetime.datetime.fromtimestamp(timestamp1)
# time_.replace(year=2020, month=1, day=1)
# time_ = time_.replace(year=2025, month=1, day=1)
# time_ = time_.strftime('%Y-%m-%d %H:%M:%S')
# time_

#%%
##########################
# 加权平均价格——画图
# Tapes:  wavg_price diagram
df_1 = agg_wavg_time
plt.figure(figsize=(12, 6))
plt.plot(df_1['time'], df_1['price_diff1'], label='Tapes')
plt.legend(loc='best')
plt.ylabel('weighted_avg_price diff_1')
plt.title('Time Series - Differenced(1) Weighted Average Price')
plt.show()

#%%
##########################
# 画出自相关ACF和偏自相关PACF图像
## acf_values = np.correlate(y, y, mode='full')
## lags = len(acf_values) - 1

lags = 40
y = (agg_wavg_time[['time', 'price_diff1']].dropna(inplace=False).set_index('time', inplace=False))
plt.figure(figsize=(12, 6))
plot_acf(y, lags=lags)  # ACF图
plt.title('ACF of Time Series')
plt.show()
plot_pacf(y, lags=lags)  # PACF图
plt.title('PACF of Time Series')
plt.show()

# #%%
# ## fit
# model = ARIMA(y, order=(1, 0, 0))
# model_fit = model.fit()
#
# #%%
# # Check whether the residuals残差 are white noise (i.e. uncorrelated)
# # 模型检测———— 是否有白噪声(自相关性)
# residuals = model_fit.resid
# plot_acf(residuals)
# plt.title('Residuals of train_wavg AR')
# plt.show()
#
# #%%
# ## forecast-- 20 steps
# ## 预测未来20个时间点
# forecast = model_fit.forecast(steps=20)
# print(forecast)
#

#%%
##############################################################################
##############################################################################
# 使用 auto_arima 自动寻找最佳 ARIMA 模型
dataset = agg_wavg_time[['time', 'price_diff1']].dropna(inplace=False).set_index('time', inplace=False)
## inplace = False 在原数据上进行

model_auto = pm.auto_arima(dataset, start_p=0, start_q=0,
                      test='adf',  # 使用ADF测试确定'd'
                      max_p=3, max_q=3,  # 设置p和q的最大值
                      m=1,  # 数据的季节性周期
                      d=None,  # 让模型自动确定最优的d
                      seasonal=False,  # 数据不包含季节性成分
                      stepwise=True,  # 使用逐步算法
                      suppress_warnings=True,  # 抑制警告信息
                      information_criterion='aic',  # 使用AIC选择最佳模型
                      trace=True)  # 打印搜索过程
model_auto_fit = model_auto.fit
# 输出模型摘要
print(model_auto.summary())

# ### about the warning :
# Warnings:
# [1] Covariance matrix calculated using the outer product of gradients (complex-step).
# 在自动ARIMA模型拟合过程中，协方差矩阵是通过梯度的外积（复步法）
# 意味着模型的参数估计是在较高的数值准确度下完成的
##协方差矩阵----衡量变量之间线性相关性。
## 在ARIMA模型中，协方差矩阵用于评估模型参数的不确定性和可信度。
## 外积梯度（OPG）----一种计算协方差矩阵的方法,适用于非线性模型的参数估计。
## 复步法--数值微分方法，用于提高参数估计的精度，可以减少计算中的数值错误。

##############################################################################
#%%
## 预测 Forecast  n_periods个点
n_periods = 8000
fc, confint = model_1.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(len(wavg_tapesprice['weighted_avg_price']), len(wavg_tapesprice['weighted_avg_price'])+n_periods)

##############################################################################
#%%
# 绘图  plotting
# 数据  make series
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

#%%

# 读取文件
column_names = ['timestamp', 'price', 'quantity']
file_path = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/Tapes/UoB_Set01_2025-01-03tapes.csv'
tapesprice_new = pd.read_csv(file_path, header=None, names=column_names, usecols=[0, 1, 2])
wavg_tapesprice_new = wavg(tapesprice_new)

#%%
##
plt.plot(wavg_tapesprice_new['weighted_avg_price'].iloc[:18000])
plt.plot(fc_series, color='red', label='fc_Tapes')
plt.fill_between(lower_series.index,
                 lower_series,
                 upper_series,
                 color='k', alpha=.15)

plt.title("Final Forecast of WWW Usage")
plt.show()