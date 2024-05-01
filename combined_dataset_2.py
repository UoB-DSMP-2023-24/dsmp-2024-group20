#%%
import os
import glob

import numpy as np
import pandas as pd
import ast
from pandas import Timedelta
import datetime

#%%
## 读入文件 read the files
def read_csvfile(file_path):
    file_name = os.path.basename(file_path)
    date = file_name.split('_')[2][:10]
    column_names = ['timestamp', 'price', 'quantity']
    data = pd.read_csv(file_path, header=None, names=column_names, usecols=[0, 1, 2])
    data['date'] = date
    return data

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
import pandas as pd
import datetime


def aggregate_Tapes(df, second_column, second):
    # 创建一个表示时间窗口的列
    df['time_window'] = (df[second_column] // second) * second
    # 计算每个交易的加权贡献
    df['weighted_price'] = df['price'] * df['quantity']

    aggregation_rules = {
        'price': ['mean', 'max', 'min'],  # 最大价格、最小价格、平均价格
        'quantity': 'sum',  # 成交数量
        'weighted_price': 'sum'  # 加权价格总和
    }
    agg_df = df.groupby('time_window').agg(aggregation_rules).reset_index()
    agg_df['weighted_avg_price'] = agg_df[('weighted_price', 'sum')] / agg_df[('quantity', 'sum')]

    # 处理众数，注意确保处理空值
    mode_prices = df.groupby('time_window')['price'].apply(lambda x: x.mode()[0] if not x.mode().empty else None)
    agg_df['most_frequent_price'] = mode_prices.values

    # 重新命名列以使输出更清晰
    agg_df.columns = ['time_window', 'mean_price', 'max_price', 'min_price', 'total_quantity',
                      'total_weighted_price', 'weighted_avg_price', 'most_frequent_price']
    agg_df['diff_meanAndWeightedPrice'] = agg_df['weighted_avg_price'] - agg_df['mean_price']

    # 删除不再需要的 total_weighted_price 列
    agg_df.drop(columns='total_weighted_price', inplace=True)

    return agg_df
# 你的循环和数据处理逻辑保持不变

# result = aggregate_data(df, 'timestamp', 3)
# print(result)


# #%%
# # test 测试  data_csv: UoB_Set01_2025-01-02tapes.csv
# directory_trytry = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/trytry/'
# data_csv, file_names = read_csvfile_all3(directory_trytry)
# ## keys: timestamp  price  quantity  (time)  (date)
#
# summary_data_csv = data_csv.describe()
# print(summary_data_csv)  ## 数据的 描述性统计数据
#
# #%%
# second = 30
# agg_df = aggregate_data(data_csv, second_column='timestamp', second=second)
#
# summary_aggdf = agg_df.describe()
# print(summary_aggdf)

#%%


input_path_Tapes = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/Tapes'

output_path = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/whole_dataset_processed'
# 获取文件夹内所有特定类型的文件，
file_paths_Tapes = glob.glob(os.path.join(input_path_Tapes, '*.csv'))  # 根据需要修改文件类型

total_Tapes = pd.DataFrame()
n=0
second = 300
date = []
# 遍历文件路径列表，处理每个文件
for file_path in file_paths_Tapes:
    original_file_name = os.path.basename(file_path)
    print("--------------Processing Tapes files:", original_file_name,"--------------")
    data_Tapes= read_csvfile(file_path)
    agg_Tapes = aggregate_Tapes(data_Tapes, second_column='timestamp', second=second)
    date = pd.to_datetime(original_file_name[10:-8])
    # date = original_file_name[0][-19:-9]
    agg_Tapes['time'] = agg_Tapes['time_window'].apply(lambda x: date + datetime.timedelta(seconds=x))
    agg_Tapes['time'] += Timedelta(hours=8)  # 需要加载包 from pandas import Timedelta
    agg_Tapes.insert(0, 'time', agg_Tapes.pop('time'))
    agg_Tapes['date'] = date
    total_Tapes = pd.concat([total_Tapes, agg_Tapes], ignore_index=True)
#%%
new_file_name = 'total_Tapes_300s.csv'
# 3. 构建新的文件路径
output_file_path = os.path.join(output_path, new_file_name)
# 处理完毕后的DataFrame可以进行保存或其他操作
# 例如，保存到CSV文件
total_Tapes.to_csv(output_file_path, index=False)

print("--------------All Tapes files processed successfully!----------------")