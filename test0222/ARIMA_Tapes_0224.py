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


#%%
## 读入文件 read the files
def read_csvfile_all3(directory, start_time=[8, 0, 0], AddtheTime=False, AddtheDate=False):
    dataframes = []
    file_names = os.listdir(directory)
    column_names = ['timestamp', 'price', 'quantity']
    for file_name in file_names:
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory, file_name)
            data = pd.read_csv(file_path, header=None, names=column_names, usecols=[0, 1, 2])
            if AddtheTime != False:
                time_date = file_name[0][10:20]
                data['time'] = pd.to_datetime(data['timestamp'], unit='s')
                data['time'] = data['time'].apply(lambda x: x.replace(year= int(time_date[:4]), month= int(time_date[5:7]), day= int(time_date[8:])))
                data['time'] += Timedelta(hours= start_time[0], minutes= start_time[1], seconds= start_time[2])
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
directory_trytry = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/trytry/'
data_csv, file_names = read_csvfile_all3(directory_trytry)
## keys: timestamp  price  quantity  (time)  (date)

summary_data_csv = data_csv.describe()
summary_data_csv

#%%
## 聚合--每5秒聚合成一个时间戳
def aggregate_data(df, second_column, aggregation_rules,second):
    df['time_window'] = (df[second_column] // second) * second
    agg_df = df.groupby('time_window').agg(aggregation_rules).reset_index()
    return agg_df

aggregation_rules = {'price': 'mean'}
second = 5
agg_df = aggregate_data(data_csv, second_column='timestamp', aggregation_rules=aggregation_rules, second=second)

add_df2 = aggregate_data(data_csv, second_column='timestamp', aggregation_rules=aggregation_rules, second=second)
#%%
## ADF test
# ADF 测试(ADF(Augmented Dickey-Fuller) 强迪基-福勒检验)——时间序列的平稳性
def ADFtest(timeseries):
    # 执行Augmented Dickey-Fuller测试
    print('Results of Augmented Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

adf_agg = ADFtest(agg_df['price'])
### Results of Augmented Dickey-Fuller Test:
# Test Statistic                   -1.952236
# p-value                           0.307900
# #Lags Used                       34.000000
# Number of Observations Used    5874.000000
# Critical Value (1%)              -3.431464
# Critical Value (5%)              -2.862032
# Critical Value (10%)             -2.567032
# dtype: float64

#%%
## --> Defference
agg_df['diff1'] = agg_df['price'].diff()
agg_df['diff2'] = agg_df['price'].diff().diff()

ADFtest(agg_df['diff1'].dropna())
ADFtest(agg_df['diff2'].dropna())

###
# Results of Augmented Dickey-Fuller Test:
# Test Statistic                  -20.085583
# p-value                           0.000000
# #Lags Used                       33.000000
# Number of Observations Used    5874.000000
# Critical Value (1%)              -3.431464
# Critical Value (5%)              -2.862032
# Critical Value (10%)             -2.567032
# dtype: float64

# Results of Augmented Dickey-Fuller Test:
# Test Statistic                  -27.224039
# p-value                           0.000000
# #Lags Used                       34.000000
# Number of Observations Used    5872.000000
# Critical Value (1%)              -3.431464
# Critical Value (5%)              -2.862032
# Critical Value (10%)             -2.567032
# dtype: float64

## 选一阶差分

#%%
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

filename = ['2025-01-02']
agg_df = timestamp_to_time(agg_df, filename)


#%%
# 5s聚合价格——画图
# Tapes:  agg_5s diagram
df_1 = agg_df.set_index('time')['diff1'].dropna()
plt.figure(figsize=(12, 6))
plt.plot(df_1, label='Tapes: diff(1) aggregated price (every %d s)' % second)
plt.legend(loc='best')
plt.xlabel('time')
plt.ylabel('weighted_avg_price diff_1')
plt.title('Time Series - Differenced(1) Weighted Average Price')
plt.show()


#%%
## 图片的线不对劲
lags = 40
y = agg_df[['time', 'diff1']].dropna().set_index('time', inplace=False)
plt.figure(figsize=(12, 6))
plot_acf(y, lags=lags)  # ACF图
plt.title('ACF of Time Series')
plt.show()
plot_pacf(y, lags=lags)  # PACF图
plt.title('PACF of Time Series')
plt.show()

## 不好找p和q的值
#%%
## split the data

df_2 =agg_df[['time', 'price', 'diff1']].dropna()

train_size = int(len(df_2)*0.999)
# val_size = train_size + int(len(df_2)*0.1)

train = df_2[:train_size]
# val = df_2[train_size:val_size]
test = df_2[train_size:]

#%%
dataset = train.set_index('time', inplace=False)['diff1']

model_auto = pm.auto_arima(dataset, start_p=0, start_q=0,
                      test='adf',  # 使用ADF测试确定'd'
                      max_p=9, max_q=9,  # 设置p和q的最大值
                      m=1,  # 数据的季节性周期
                      d=None,  # 让模型自动确定最优的d
                      seasonal=False,  # 数据不包含季节性成分
                      stepwise=True,  # 使用逐步算法
                      suppress_warnings=True,  # 抑制警告信息
                      information_criterion='aic',  # 使用AIC选择最佳模型
                      trace=True)  # 打印搜索过程
# 输出模型摘要
print(model_auto.summary())

#%%
n_periods = len(test)
fc, confint = model_auto.predict(n_periods=n_periods, return_conf_int=True)
fc_df = pd.DataFrame(fc, columns=['forecast'])
index_of_fc = test['time']
fc_df = fc_df.set_axis(index_of_fc, axis='index')


#%%

##
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

##
test_ = pd.DataFrame(test.set_index('time', inplace=False)['diff1'])
plt.plot(test_['diff1'])
plt.plot(fc_df, color='red', label='fc_Tapes')
# plt.fill_between(lower_series.index,
#                  lower_series,
#                  upper_series,
#                  color='k', alpha=.15)

plt.title("Final Forecast")
plt.show()

#%%

