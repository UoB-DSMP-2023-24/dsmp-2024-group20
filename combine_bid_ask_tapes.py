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
import re
import os
import ast
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import Timedelta
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


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
## 聚合--每5秒聚合成一个时间戳
# def aggregate_data(df, second_column, aggregation_rules,second):
#     df['time_window'] = (df[second_column] // second) * second
#     agg_df = df.groupby('time_window').agg(aggregation_rules).reset_index()
#     return agg_df
# test:
# aggregation_rules = {'price': 'mean'}  ## price: the true price 真实成交价
# agg_df = aggregate_data(data_csv, second_column='timestamp', aggregation_rules=aggregation_rules, second=second)

'''
Tapes dataset:
['time_window', 'mean_price', 'max_price', 'min_price',
       'most_frequent_price', 'total_quantity', 'weighted_avg_price',
       'diff_meanAndWeightedPrice']  
       'total_quantity': 时间窗内的累计成交量
       'most_frequent_price': 成交数最多的价格————时间窗内较优价格
'''
def aggregate_data(df, second_column, second):
    # 创建一个表示时间窗口的列
    df['time_window'] = (df[second_column] // second) * second

    # 计算每个交易的加权贡献
    df['weighted_price'] = df['price'] * df['quantity']

    # 定义聚合规则：最大价格、最小价格、成交数量，以及加权价格的总和
    aggregation_rules = {
        'price': ['mean', 'max', 'min', pd.Series.mode],  # 最大价格、最小价格、成交最多的价格（众数）
        'quantity': 'sum',  # 成交数量
        'weighted_price': 'sum'  # 加权价格总和
    }

    # 对数据进行分组并应用聚合规则
    agg_df = df.groupby('time_window').agg(aggregation_rules).reset_index()

    # 计算加权均价
    agg_df['weighted_avg_price'] = agg_df[('weighted_price', 'sum')] / agg_df[('quantity', 'sum')]

    # 处理众数，确保始终返回单个值。如果众数是一个序列，则取第一个元素；如果为空，则返回None。
    agg_df[('price', 'mode')] = agg_df[('price', 'mode')].apply(
        lambda x: x[0] if isinstance(x, pd.Series) and not x.empty else x)

    # 调整列名称，使其更加清晰
    agg_df.columns = ['time_window', 'mean_price', 'max_price', 'min_price', 'most_frequent_price', 'total_quantity',
                      'total_weighted_price', 'weighted_avg_price']
    agg_df['diff_meanAndWeightedPrice'] = agg_df['weighted_avg_price']-agg_df['mean_price']

    # 删除不再需要的 total_weighted_price 列
    agg_df.drop(columns='total_weighted_price', inplace=True)

    return agg_df

# result = aggregate_data(df, 'timestamp', 3)
# print(result)


#%%
# test 测试  data_csv: UoB_Set01_2025-01-02tapes.csv
directory_trytry = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/trytry/'
data_csv, file_names = read_csvfile_all3(directory_trytry)
## keys: timestamp  price  quantity  (time)  (date)

summary_data_csv = data_csv.describe()
print(summary_data_csv)  ## 数据的 描述性统计数据

#%%
second = 30
agg_df = aggregate_data(data_csv, second_column='timestamp', second=second)

summary_aggdf = agg_df.describe()
print(summary_aggdf)

#%%
######################################################################################################################

def add_quotes_to_specific_word(s, word):
    pattern = r'\b' + re.escape(word) + r'\b'
    return re.sub(pattern, f"'{word}'", s)

## list.insert(position=2, data='new data')
def insert_date(list, position, date):
    # date = pd.to_datetime(file_name[10:20])
    # date = file_name[10:20]
    list.insert(position, date)
    return list

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
## split_txtdata from somethingaboutlist.py split_txtdata_4()
# 第一列 时间戳，第二列 bid订单[ [价格, 数量],[价格, 数量],...,[价格，数量] ]，第三列 ask定价单[[价格, 数量],...,[价格, 数量]]
def split_txtdata(dataset:list):
    timestamp = []
    bid = []
    price_bid = ()
    ask = []
    price_ask = ()
    # 提取时间戳
    for i in range(len(dataset)):
        timestamp.append(dataset[i][0])

    # 提取 价格单
    for i in range(len(dataset)):
        # 价格单中 买家给出的价格单 a buy list，只记录价格和数量
        bid.extend([dataset[i][2][0][1]])
        # 价格单中 卖家给出的价格单 a sell list
        ask.extend([dataset[i][2][1][1]])

    dataset = {'timestamp': timestamp, 'bid': bid, 'ask': ask}
    print('The type of dataset is', type(dataset))
    print('The names of the keys are : ', dataset.keys())
    return dataset


#%%
# 文件路径
file_path = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/trytry_lob/'
LOB_list, file_names = read_LOBtxt(file_path)
dataset_bidask = split_txtdata(LOB_list)

df = pd.DataFrame(dataset_bidask)

# 导出到CSV文件，这里的index=False表示不导出行索引
# df.to_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/dataset_bidask4.csv', index=False)

##########################################################################################################
##########################################################################################################
data_csv = df
def extract_values(lst):
    # 提取列表中的价格和数量，如果列表为空，返回None
    if lst:
        prices, quantities = zip(*lst)  # 解包列表中的元素
        return max(prices), min(prices), sum(quantities)
    else:
        return None, None, 0

# 假设data_csv是你的DataFrame
data_csv = pd.DataFrame({
    'timestamp': [0.000, 0.279, 1.333],
    'bid': [[], [[1, 6]], [[1, 6]]],
    'ask': [[], [], [[800, 1]]]
})

# 应用函数并创建新列  ## total_bid, total_ask 表示当前时间戳的订单的数量
data_csv[['max_bid', 'min_bid', 'total_bid']] = pd.DataFrame(data_csv['bid'].apply(lambda x: extract_values(x)).tolist(), index=data_csv.index)
data_csv[['max_ask', 'min_ask', 'total_ask']] = pd.DataFrame(data_csv['ask'].apply(lambda x: extract_values(x)).tolist(), index=data_csv.index)

# 输出加工后的DataFrame
data_csv
