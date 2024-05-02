import re
import os
import ast
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import Timedelta
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

#%%
def add_quotes_to_specific_word(s, word):
    # 使用正则表达式为特定单词添加引号
    # \b 是单词边界，确保整个单词被匹配
    # 使用 re.escape 来处理 word 中可能包含的任何特殊字符
    pattern = r'\b' + re.escape(word) + r'\b'
    return re.sub(pattern, f"'{word}'", s)

## list.insert(position=2, data='new data')
def insert_date(list, position, date):
    # date = pd.to_datetime(file_name[10:20])
    # date = file_name[10:20]
    list.insert(position, date)
    return list

# file_names[0][10:14] ## '2025'
# file_names[0][15:17] ## '01'
# file_names[0][18:20] ## '02'

# 文件路径
file_path = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/trytry_lob/'

## ----------------------------------
def read_LOBtxt(directory):
    LOB_list = []  # 用于存储
    file_names = os.listdir(directory)  # 获取目录中的所有文件名  # import os
    # column_names = ['timestamp', 'price', 'quantity']
    for file_name in file_names:
        if file_name.endswith('.txt'):  # 先检查是否是 txt 文件
            file_path = os.path.join(directory, file_name)  # 正确地构建文件路径
            with open(file_path, 'r') as file:
                txt = file.read()
                data = add_quotes_to_specific_word(txt, 'Exch0')
                lines = data.split('\n')
                # LOB_list = [ast.literal_eval(line) for line in lines if line]  # import ast
                for line in lines:
                    if line:
                        each_line = ast.literal_eval(line)  # 转换行
                        # 在转换后的列表中插入日期信息
                        date = file_name[10:20]
                        line_withdate = insert_date(each_line, 3, date)  # 假设在列表的末尾插入日期
                        LOB_list.append(line_withdate)
    return LOB_list, file_names



#%%
## save the List
### LOB_list, file_names_lob = read_LOBtxt(file_path)
LOB_list, file_names = read_LOBtxt(file_path)


#%%
## ----------------------------------
## output the List that we have read
## 输出读取的列表文件
# import csv
# with open("LOB_list.csv", "w", newline='') as file:
#     # 创建一个 CSV writer 对象
#     csv_writer = csv.writer(file)
#     # 写入列表中的数据行
#     csv_writer.writerows(LOB_list)

#### df_bid.to_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/dataset5_bid.csv', index=False)

#%%
## ---------------------------------------------
###  this function is "split_txtdata_5(dataset:list)" in somethingaboutlist.py


## 调试 dateset['data']， ‘data’ 没有输出。
def split_txtdata(dataset:list, filenames):
    date = []
    date_len = 0
    timestamp_bid = []
    timestamp_ask = []
    bid_price = []
    bid_quantity = []
    bid = []
    ask_price = []
    ask_quantity = []
    ask = []
    for i in range(len(dataset)):   ## 按每条list数据处理
        for j in range(len(dataset[i][2][0][1])):
            timestamp_bid.append(dataset[i][0])
            bid.append(dataset[i][2][0][1][j])
            bid_price.append(dataset[i][2][0][1][j][0])
            bid_quantity.append(dataset[i][2][0][1][j][1])
        for q in range(len(dataset[i][2][1][1])):
            timestamp_ask.append(dataset[i][0])
            ask.append(dataset[i][2][1][1][q])
            ask_price.append(dataset[i][2][1][1][q][0])
            ask_quantity.append(dataset[i][2][1][1][q][1])
    for filename in filenames:
        date_len = 0
        for i in range(len(dataset)):
            if dataset[i][3] == filename[10:20]:
                date_len += 1
        date.extend([filename[10:20]] * date_len)

    dataset = {"date": date, 'bid': bid, 'timestamp_bid': timestamp_bid, 'bid_price': bid_price, 'bid_quantity': bid_quantity, 'ask': ask, 'timestamp_ask': timestamp_ask, 'ask_price': ask_price, 'ask_quantity': ask_quantity}
    print('The type of dataset is', type(dataset))
    print('The names of the keys are : ', dataset.keys())
    return dataset

dataset_bidask = split_txtdata(LOB_list, file_names)

#%%
## 将dataset_bidask 转化成两个 数据框

def bidaskList_to_dataframe(dataset_bidask):
    date_ask = []
    date_bid = []
    keys_bid = ['timestamp_bid', 'bid_price', 'bid_quantity']
    keys_ask = ['timestamp_ask', 'ask_price', 'ask_quantity']
    ask_dict = {key: dataset_bidask[key] for key in keys_ask if key in dataset_bidask}
    bid_dict = {key: dataset_bidask[key] for key in keys_bid if key in dataset_bidask}
    # ## 是否增加一列date？ 暂时不增加？
    df_bid = pd.DataFrame(data=bid_dict)
    df_ask = pd.DataFrame(data=ask_dict)
    return df_bid, df_ask

# keys_bid = ['timestamp_bid','bid_price','bid_quantity']
# bid_dict = {key: dataset_bidask5[key] for key in keys_bid if key in dataset_bidask5}
# df_bid = pd.DataFrame(data=bid_dict)
#
# keys_ask = ['timestamp_ask', 'ask_price', 'ask_quantity']
# ask_dict = {key: dataset_bidask5[key] for key in keys_ask if key in dataset_bidask5}
# df_ask = pd.DataFrame(data=ask_dict)
#
# # 导出到CSV文件，这里的index=False表示不导出行索引
# df_bid.to_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/dataset5_bid.csv', index=False)
# df_ask.to_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/dataset5_ask.csv', index=False)


#%%
df_bid, df_ask = bidaskList_to_dataframe(dataset_bidask)

#%%
## 直接读取现有的数据
data_ask = pd.read_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/LOB_processed/LOB_ask/2025-01-02.csv')
data_bid = pd.read_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/LOB_processed/LOB_bid/2025-01-02.csv')

#%%
## calculate the weighted_price
### wavg_price: weight average price based on quantity
def wavg(data_csv, timestamp, price, quantity):
    '''

    :param data_csv: dataframe
    :param timestamp: the column's name
    :param price: the column's name
    :param quantity: the column's name
    :return:
    '''
    ### or using keys() to get these column's names
    df = pd.DataFrame()
    data_csv['wavg_plus_price'] = data_csv[price] * data_csv[quantity]
    df = data_csv.groupby([timestamp]).agg({'wavg_plus_price': 'sum', quantity: 'sum'}).reset_index()
    df['wavg_price'] = df['wavg_plus_price'] / df[quantity]
    return df

# df_bid.keys()
## (['timestamp_bid', 'bid_price', 'bid_quantity'], dtype='object')
# df_ask.keys()
## (['timestamp_ask', 'ask_price', 'ask_quantity'], dtype='object')

wavg_bid = wavg(df_bid, 'timestamp_bid', 'bid_price', 'bid_quantity')
wavg_ask = wavg(df_ask, 'timestamp_ask', 'ask_price', 'ask_quantity')

wavg_bid2 = wavg(data_bid, 'timestamp', 'price', 'quantity')
wavg_ask2 = wavg(data_ask, 'timestamp', 'price', 'quantity')
# wavg_bid.keys()
# Out[151]: Index(['timestamp_bid', 'wavg_plus_price', 'bid_quantity', 'wavg_price'], dtype='object')

#%%

## Aggregation Data
def aggregate_data(df, second_column, aggregation_rules,second):
    df['time_window'] = (df[second_column] // second) * second
    aggregated_df = df.groupby('time_window').agg(aggregation_rules).reset_index()
    return aggregated_df

aggregation_rules = {'wavg_price': 'mean'}
agg_bid = aggregate_data(wavg_bid, second_column='timestamp_bid', aggregation_rules=aggregation_rules, second=5)
agg_ask = aggregate_data(wavg_ask, second_column='timestamp_ask', aggregation_rules=aggregation_rules, second=5)

agg_bid2 = aggregate_data(wavg_bid2, second_column='timestamp', aggregation_rules=aggregation_rules, second=5)
agg_ask2 = aggregate_data(wavg_ask2, second_column='timestamp', aggregation_rules=aggregation_rules, second=5)
# agg_bid.keys()
# Out[153]: Index(['time_window', 'wavg_price'], dtype='object')

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




#%%
## Converts a timestamp to a time
## 将时间戳转化成时间

## The London Stock Exchange opens at 8:00 a.m. UK time and closes at 4:30 p.m.
### with the market closed from 12:00 to 12:02 noon.
## 伦敦股票交易市场：8:00开盘,16:30收盘。12:00-12:02是休息时间。每天交易时长为8小时28分钟。
## default: start_time = [8, 0, 0]

## 函数需要与 原本的 file_name 中的日期 相关联; 或者只转化时分秒，不增加日期数据
## filenames 表示 file_names（上面的读取LOB txt文件的函数）

# def timestamp_to_time(data, filenames, start_time=[8, 0, 0]):
#     """
#     Converts a timestamp to a time
#     :param data: pandas dataframe with timestamps 【！】 timestamp column's name : "time_window"
#     :param filenames: come from the read_csvfile_all3()
#     :param start_time: list; default value [8, 0, 0]
#     :return: pandas dataframe with time: YYYY-MM-DD HH:mm:ss.ms
#     """
#     for filename in filenames:
#         data['time'] = pd.to_datetime(data['time_window'], unit='s')
#         data['time'] = data['time'].apply(lambda x: x.replace(year=int(filename[10:14]),
#                                                               month=int(filename[15:17]),
#                                                               day=int(filename[18:20])))
#         data['time'] += Timedelta(hours=start_time[0], minutes=start_time[1], seconds=start_time[2])
#
#     return data

### agg_bid_time = timestamp_to_time(agg_bid, file_names)



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


adf_result_bid = ADFtest(agg_bid['wavg_price'])
adf_result_ask = ADFtest(agg_ask['wavg_price'])

adf_result_bid2 = ADFtest(agg_bid2['wavg_price'])
adf_result_ask2 = ADFtest(agg_ask2['wavg_price'])
## bid-- 不需要差分
# Results of Augmented Dickey-Fuller Test:
# Test Statistic                  -19.382527
# p-value                           0.000000
# #Lags Used                        8.000000
# Number of Observations Used    6111.000000
# Critical Value (1%)              -3.431421
# Critical Value (5%)              -2.862013
# Critical Value (10%)             -2.567022
# dtype: float64

## ask-- 不需要差分
# Results of Augmented Dickey-Fuller Test:
# Test Statistic                -1.132689e+01  这里e+01是10^1
# p-value                        1.137648e-20
# #Lags Used                     1.900000e+01
# Number of Observations Used    6.100000e+03
# Critical Value (1%)           -3.431422e+00
# Critical Value (5%)           -2.862014e+00
# Critical Value (10%)          -2.567022e+00
# dtype: float64

## 直接读取的bid文件不需要差分
# Results of Augmented Dickey-Fuller Test:
# Test Statistic                -7.674703e+00
# p-value                        1.559954e-11
# #Lags Used                     3.400000e+01
# Number of Observations Used    6.085000e+03
# Critical Value (1%)           -3.431425e+00
# Critical Value (5%)           -2.862015e+00
# Critical Value (10%)          -2.567023e+00
# dtype: float64

## 直接读取的ask文件不需要差分
# Results of Augmented Dickey-Fuller Test:
# Test Statistic                -1.219069e+01   这里科学计数法 e+01
# p-value                        1.287862e-22
# #Lags Used                     2.000000e+01
# Number of Observations Used    6.099000e+03
# Critical Value (1%)           -3.431423e+00
# Critical Value (5%)           -2.862014e+00
# Critical Value (10%)          -2.567022e+00
# dtype: float64
#%%
def timestamp_to_time(data, filenames, start_time=[8, 0, 0]):
    """
    Converts a timestamp to a time
    :param data: pandas dataframe with timestamps 【！】 timestamp column's name : "time_window"
    :param filenames: come from the list of dates
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

agg_bid2 = timestamp_to_time(agg_bid2, filename)
agg_ask2 = timestamp_to_time(agg_ask2, filename)

#%%
## ACF & PACF 图像
## 记得及时更新画图的模块！！
agg_df = agg_bid2
# agg_df = agg_ask2

lags = 30

y = agg_df[['time_window', 'wavg_price']].dropna().set_index('time_window', inplace=False)
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

df_2 =agg_bid2.dropna()
# df_2 =agg_ask2.dropna()


train_size = int(len(df_2)*0.99)
# val_size = train_size + int(len(df_2)*0.1)

train = df_2[:train_size]
# val = df_2[train_size:val_size]
test = df_2[train_size:]



#%%
dataset = train.set_index('time', inplace=False)['wavg_price']

#%%
model_auto = pm.auto_arima(dataset, start_p=0, start_q=0,
                      test='adf',  # 使用ADF测试确定'd'
                      max_p=5, max_q=5,  # 设置p和q的最大值
                      m=1,  # 数据的季节性周期
                      # d=None,  # 让模型自动确定最优的d
                      seasonal=False,  # 数据不包含季节性成分
                      stepwise=False,  # 使用逐步算法？
                      suppress_warnings=True,  # 抑制警告信息
                      information_criterion='aic',  # 使用AIC选择最佳模型
                      trace=True)  # 打印搜索过程
# 输出模型摘要
print(model_auto.summary())

### AIC和BIC值太高了———————— 绝对值意义不大， 要看不同模型之间的相对值
### 需要尝试重新处理数据?  ## 后面已经重新对数据进行了处理
### 要不尝试给数据加上 年月日 时分秒？ ## 后面已经做了改进
#%%
## forecast
n_periods = len(test)
fc, confint = model_auto.predict(n_periods=n_periods, return_conf_int=True)
fc_df = pd.DataFrame(fc, columns=['forecast'])
index_of_fc = test['time']
fc_df = fc_df.set_axis(index_of_fc, axis='index')

#%%
# ## 调整轴
# lower_series = pd.Series(confint[:, 0], index=index_of_fc)
# upper_series = pd.Series(confint[:, 1], index=index_of_fc)

#%%
test.set_index('time', inplace=True)
train.set_index('time', inplace=True)
#%%
plt.plot(train['wavg_price'][-800:], label='train')
plt.plot(test['wavg_price'], label='test', color='orange')
plt.plot(fc_df, color='green', linestyle='--', label='Forecast_Tapes')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('ARIMA Forecast vs True Values')
plt.legend()
plt.show()


