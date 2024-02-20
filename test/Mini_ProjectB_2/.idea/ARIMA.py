# 预加载模块
import os
import re
import ast
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# ————————————————————————————————————————————————
# 读取数据
#%%
## 给 Exch0 增加引号, 函数需要re模块
def add_quotes_to_word(s, word):
    pattern = r'\b' + re.escape(word) + r'\b'
    return re.sub(pattern, f"'{word}'", s)

## directory 最后需要以'/'结尾， 函数需要os, pandas, ast模块
def read_txtfile_all(directory, special_word):
    data_list = []
    file_names = os.listdir(directory)
    for filename in file_names:
        path = os.path.join(directory, filename)
        with open(path, 'r') as file:
            data_str = file.read()
            data_str = add_quotes_to_specific_word(data_str, special_word)

    # 分割数据并转换成列表
    lines = data_str.split('\n')
    data_list = [ast.literal_eval(line) for line in lines if line]
    return data_list

#%%
# ---------------------------------------------------------------
#%%
## 完全展开每一个订单[price, quantity], 输出字典型数据。
## 可以将输出结果分成两个数据框，一个bid, 一个ask。
def split_txtdata(dataset: list):
    timestamp_bid = [data[0] for data in dataset for _ in range(len(data[2][0][1]))]
    bid = [bid_data for data in dataset for bid_data in data[2][0][1]]
    bid_price = [data[0] for data in bid]
    bid_quantity = [data[1] for data in bid]

    timestamp_ask = [data[0] for data in dataset for _ in range(len(data[2][1][1]))]
    ask = [ask_data for data in dataset for ask_data in data[2][1][1]]
    ask_price = [data[0] for data in ask]
    ask_quantity = [data[1] for data in ask]

    # 构建最终的数据集字典
    dataset = {
        'bid': bid,
        'timestamp_bid': timestamp_bid,
        'bid_price': bid_price,
        'bid_quantity': bid_quantity,
        'ask': ask,
        'timestamp_ask': timestamp_ask,
        'ask_price': ask_price,
        'ask_quantity': ask_quantity
    }

    print('The type of dataset is', type(optimized_dataset))
    print('The names of the keys are : ', optimized_dataset.keys())

    return dataset
#%%
# 测试 test
directory = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/trytry_lob/'
data_txt0218 = read_txtfile_all(directory, "Exch0")
data_txt0218 = split_txtdata(data_txt0218)

#%%
# —————————————————————————————————————————————————
# 数据框结构的数据 Type: turn to DataFrame
## 下面的代码 暂时使用测试数据 data_txt0218
#%%
def dirtodf(dir_data):
    keys_bid = ['timestamp_bid', 'bid_price', 'bid_quantity']
    keys_ask = ['timestamp_ask', 'ask_price', 'ask_quantity']
    bid_dict = {key: dir_data[key] for key in keys_bid if key in dir_data}
    df_bid = pd.DataFrame(data=bid_dict)
    ask_dict = {key: dir_data[key] for key in keys_ask if key in dir_data}
    df_ask = pd.DataFrame(data=ask_dict)
    return df_bid, df_ask
#%%
# 测试 test
df_bid, df_ask = dirtodf(data_txt0218)

# # 导出到CSV文件，这里的index=False表示不导出行索引
# df_bid.to_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/data_txt0218_bid.csv', index=False)
# df_ask.to_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/data_txt0218_ask.csv', index=False)
#%%
# 将同一个时间戳的订单价格 进行加权平均， 得到加权平均价格 weighted average price （wavg_price）
df_bid['wavg_bidprice'] = df_bid['bid_price'] * df_bid['bid_quantity']
wavg_bidprice = df_bid.groupby(['timestamp_bid']).agg({'wavg_bidprice': 'sum', 'bid_quantity': 'sum'}).reset_index()
wavg_bidprice['weighted_avg_price'] = wavg_bidprice['wavg_bidprice']/wavg_bidprice['bid_quantity']

# df_ask 数据框 ： 计算ask订单的加权价格
df_ask['wavg_askprice'] = df_ask['ask_price'] * df_ask['ask_quantity']
wavg_askprice = df_ask.groupby(['timestamp_ask']).agg({'wavg_askprice': 'sum', 'ask_quantity': 'sum'}).reset_index()
wavg_askprice['weighted_avg_price'] = wavg_askprice['wavg_askprice']/wavg_askprice['ask_quantity']


#%%
# ADF 测试(ADF(Augmented Dickey-Fuller) 强迪基-福勒检验)——时间序列的平稳性
def ADFtest(timeseries):
    # 执行Augmented Dickey-Fuller测试
    print('Results of Augmented Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
#%%
# 测试 加权时间序列bid数据的平稳性
adf_result_bid = ADFtest(wavg_bidprice['weighted_avg_price'])
adf_result_ask = ADFtest(wavg_askprice['weighted_avg_price'])

## 结果 result
# -1- adf_result_bid
# Results of Augmented Dickey-Fuller Test:
# Test Statistic                    -36.203467
# p-value                             0.000000
# #Lags Used                         75.000000
# Number of Observations Used    332700.000000
# Critical Value (1%)                -3.430370
# Critical Value (5%)                -2.861549
# Critical Value (10%)               -2.566775
# dtype: float64
########## 检验统计量明显小于三个临界值 & P值很小 ——有强有力的证据说明，bid的加权价格具有平稳性
# -2- adf_result_ask
# Results of Augmented Dickey-Fuller Test:
# Test Statistic                    -39.608938
# p-value                             0.000000
# #Lags Used                         87.000000
# Number of Observations Used    332645.000000
# Critical Value (1%)                -3.430370
# Critical Value (5%)                -2.861549
# Critical Value (10%)               -2.566775
# dtype: float64
########## 检验统计量明显小于三个临界值 & P值很小 ——有强有力的证据说明，bid的加权价格具有平稳性


#%%
# Bid and Ask wavg_price diagram
plt.figure(figsize=(12, 6))
plt.plot(wavg_bidprice['weighted_avg_price'], label='Bid')
plt.plot(wavg_askprice['weighted_avg_price'], label='Ask')
plt.legend(loc='best')
plt.title('Time Series')
plt.show()


#%%
# ——————————————————————————————————————————————————
# 进行差分，利用ADF test 选取最佳阶数
## 由ADF检验结果——拒绝原假设->我们可以认为时间序列数据具有平稳性。
## 所以不进行差分步骤

# ——————————————————————————————————————————————————
#



